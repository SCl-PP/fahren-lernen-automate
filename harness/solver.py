"""Fahren Lernen screen triage + question solvers.

Three Gemini calls live here:
- triage_screen(): classify the screen (foto/text/film/other), locate play button
- solve_question(): standard foto/text question -> answer indices + checkbox bboxes
- solve_film_question(): post-playback analysis with the recorded video in context
"""
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .capture import Capture
from .decide import GeminiClient


class BoundingBox(BaseModel):
    """Normalized bounding box in Gemini's trained 0-1000 coordinate space."""

    ymin: int = Field(description="Obere Kante, 0-1000 normalisiert.")
    xmin: int = Field(description="Linke Kante, 0-1000 normalisiert.")
    ymax: int = Field(description="Untere Kante, 0-1000 normalisiert.")
    xmax: int = Field(description="Rechte Kante, 0-1000 normalisiert.")

    def center_px(self, image_w: int, image_h: int) -> tuple[int, int]:
        cx = int((self.xmin + self.xmax) / 2 / 1000 * image_w)
        cy = int((self.ymin + self.ymax) / 2 / 1000 * image_h)
        return cx, cy


class AnswerBox(BoundingBox):
    index: int = Field(description="1-basierter Index der Antwortoption, von oben.")
    label: str = Field(description="Der Text der Antwortoption.")


class FahrschulAntwort(BaseModel):
    explanation: str = Field(
        description=(
            "Schritt-fuer-Schritt-Analyse im Fahrlehrer-Modus: "
            "(1) Situation und Frage WOERTLICH erfassen (achte auf 'muss' vs 'darf', "
            "'immer' vs 'nur', konkrete Werte, Fahrzeugklassen). "
            "(2) Zugrundeliegende Regel benennen (§StVO wenn moeglich, Verkehrszeichen, "
            "physikalisches Prinzip). "
            "(3) JEDE Antwortoption EINZELN pruefen mit Begruendung, warum sie korrekt "
            "oder falsch ist. Pruefe auch Optionen, die intuitiv falsch wirken -- oft "
            "sind sie trotzdem Teil der Musterloesung. "
            "(4) Finale Zusammenfassung welche Indizes zur offiziellen Musterloesung "
            "gehoeren und warum. "
            "Diese Analyse ZUERST ausfuehren; correct_solutions daraus ableiten."
        )
    )
    correct_solutions: list[int] = Field(
        description=(
            "Alle korrekten Antwort-Indizes aus der Analyse in explanation, 1-basiert, "
            "aufsteigend sortiert. Mehrere richtige Antworten sind in der deutschen "
            "Theorie-Pruefung der Normalfall; eine HALBRICHTIGE Auswahl (zu wenige "
            "oder zu viele Haeckchen) gilt als FALSCH. Gib lieber eine zu viel als "
            "eine Pflicht-Loesung zu wenig an."
        )
    )
    answer_boxes: list[AnswerBox] = Field(
        description="Alle sichtbaren Antwortoptionen mit Checkbox-Bounding-Boxes."
    )


class FahrschulInputAntwort(BaseModel):
    """Answer for a question that expects a typed value (number or short text)."""

    explanation: str = Field(
        description=(
            "Schritt-fuer-Schritt-Analyse: (1) Was wird genau gefragt? Einheit? "
            "Fahrzeugklasse? Strassentyp? "
            "(2) Welche Regel/Formel/Wert gilt (§StVO, Faustformel, Verkehrsschild)? "
            "(3) Finaler Wert mit knapper Begruendung. Falls mehrere Werte in der "
            "Frage genannt sind, erklaere welcher der gefragte ist. "
            "Diese Analyse ZUERST; answer_value daraus ableiten."
        )
    )
    answer_value: str = Field(
        description=(
            "Der EXAKT in das Eingabefeld zu tippende Wert (z.B. '130' oder 'Blinker'). "
            "Keine Einheit einschliessen, falls die Einheit (z.B. 'km/h', 'm', 's') "
            "im UI bereits separat steht. Nur die reine Antwort."
        )
    )
    input_field: BoundingBox = Field(
        description=(
            "Bounding Box des Eingabefelds (0-1000) -- der Antwortkasten mit "
            "Placeholder wie 'Antwort', in den der Wert getippt werden soll."
        )
    )


class ButtonLocation(BaseModel):
    """Result of looking for a specific labeled button on screen."""

    found: bool = Field(description="True wenn der Button sichtbar ist.")
    box: BoundingBox | None = Field(
        default=None,
        description="Bounding Box des Buttons (0-1000). Null wenn nicht gefunden.",
    )
    reasoning: str = Field(description="Kurz: warum gefunden oder nicht?")


class ScreenTriage(BaseModel):
    question_type: Literal["foto", "text", "film", "input", "other"] = Field(
        description=(
            "foto=statische Bildfrage mit Checkboxen, text=reine Textfrage mit Checkboxen, "
            "film=Videofrage mit Play-Button, input=Frage mit Texteingabefeld (z.B. "
            "Antwortkasten fuer Zahl + 'km/h'), other=sonstiger Screen (Menu, Loesung)."
        )
    )
    reasoning: str = Field(description="Kurz: woran erkannt?")
    video_playing: bool = Field(
        description="True wenn ein Video gerade abspielt. Sonst false."
    )
    video_start_button: BoundingBox | None = Field(
        default=None,
        description=(
            "Bounding Box des 'Film starten'/Play-Buttons. "
            "Nur gesetzt wenn question_type=film und video_playing=false."
        ),
    )


_INSTRUCTOR_SYSTEM = (
    "Du bist ein erfahrener, hoch qualifizierter Fahrlehrer in Deutschland "
    "mit profundem Wissen der Strassenverkehrsordnung (StVO), der Fahrerlaubnis-"
    "Verordnung (FeV), aller Verkehrszeichen und Zusatzzeichen (StVO-Anlagen 1-4), "
    "Fahrzeugtechnik, Verkehrsphysik und typischer Gefahrensituationen. "
    "Du hast tausende Fragen der offiziellen deutschen Theorie-Pruefung gesehen und "
    "beantwortest sie mit MUSTERLOESUNGS-Genauigkeit.\n\n"
    "Methodik pro Frage:\n"
    "1. Lies die Frage WORTWOERTLICH. Entscheidende Schluesselwoerter:\n"
    "   - 'muss' (Pflicht) vs 'darf' (Erlaubnis) vs 'soll' (Empfehlung)\n"
    "   - 'immer', 'stets', 'grundsaetzlich' vs 'nur', 'manchmal'\n"
    "   - konkrete Zahlen, Verkehrszeichen, Fahrzeugklassen, Strassentyp\n"
    "2. Identifiziere die zugrundeliegende Regel:\n"
    "   - §StVO-Paragraph wenn moeglich benennen\n"
    "   - Verkehrszeichen aus den StVO-Anlagen\n"
    "   - Physikalische Formel (Reaktionsweg = v/10*3, Bremsweg = (v/10)^2, "
    "     Anhalteweg = Reaktionsweg + Bremsweg)\n"
    "3. Pruefe JEDE Antwortoption EINZELN gegen die Regel:\n"
    "   - Ist sie faktisch korrekt?\n"
    "   - Trifft sie in der genau beschriebenen Situation zu?\n"
    "   - Pruefe auch Optionen, die intuitiv falsch wirken -- oft sind sie "
    "     trotzdem Teil der Musterloesung.\n"
    "4. Entscheide erst dann, welche Indizes die Musterloesung bilden.\n\n"
    "Wichtige Faustregeln und Richtwerte:\n"
    "- Innerorts 50 km/h; Ausserorts 100 km/h; Autobahn Richtgeschwindigkeit 130 km/h\n"
    "- Mindestabstand ausserorts: halber Tacho in Metern bzw. mindestens 2 Sekunden\n"
    "- Reaktionsweg (m) = (v in km/h) / 10 * 3\n"
    "- Bremsweg trocken (m) = (v / 10)^2; Gefahrenbremsung: (v/10)^2 / 2\n"
    "- Promillegrenze 0.5 (0.0 fuer Fahranfaenger/unter 21)\n"
    "- Mehrere richtige Antworten sind der NORMALFALL; die halbrichtige Auswahl "
    "  gilt in der Pruefung als falsch.\n"
    "- Antwortoptionen mit 'immer/nie/grundsaetzlich' sind oft FALSCH (StVO-Ausnahmen).\n"
    "- Vorsichtiges, gefahrenbewusstes und defensives Verhalten ist oft RICHTIG.\n\n"
    "Lass dich NIEMALS von plausibel klingenden, aber unvollstaendigen oder falschen "
    "Antworten irrefuehren. Wenn du schwankst, waehle die Option, die der strengeren "
    "Auslegung der Verkehrsregel entspricht.\n\n"
    "KRITISCH -- Pruefungsdenken statt Alltagsdenken:\n"
    "Diese Fragen stammen aus einer formalen THEORIE-PRUEFUNG. Du bewertest nach "
    "der offiziellen MUSTERLOESUNG des Pruefungsgremiums, nicht nach gesundem "
    "Menschenverstand oder praktischer Fahrererfahrung.\n\n"
    "Haeufige Distraktor-Antworten, die plausibel klingen aber in der Pruefung "
    "FALSCH sind:\n"
    "- Tangentiale Begruendungen: 'spart Zeit', 'spart Sprit', 'ist hoeflich', "
    "  'nicht viel Gewinn' -- Nebeneffekte, keine StVO-Begruendung\n"
    "- 'Common Sense'-Antworten ohne konkreten §StVO- oder Gefahrenbezug\n"
    "- Partiell wahre Antworten, die aber nicht die PRIMAERE Gefahr/Regel adressieren\n"
    "- Antworten, die praktisches Fahrerverhalten beschreiben, aber nicht die "
    "  formale Verkehrssicherheits-Begruendung treffen\n\n"
    "Die Musterloesung stuetzt sich fast IMMER auf:\n"
    "- Eine konkrete StVO-Vorschrift (z.B. §1 Ruecksicht, §3 Geschwindigkeit, "
    "  §4 Abstand, §5 Ueberholen, §8 Vorfahrt, §9 Abbiegen, §10 Einfahren)\n"
    "- Eine direkte physikalische/technische Gefahr (Bremsweg, toter Winkel, "
    "  Sichtweite, Fahrbahnhaftung)\n"
    "- Eine Verkehrssicherheits-Maxime: Unfallvermeidung, Schutz schwaecherer "
    "  Verkehrsteilnehmer (Fussgaenger, Radfahrer, Kinder), defensives Fahren\n\n"
    "Fragen-Lesestrategie: wenn die Frage 'Warum tun Sie X?' lautet, ist die "
    "gesuchte Antwort fast immer eine SICHERHEITS- oder GESETZES-Begruendung, "
    "KEINE Komfort- oder Zeit-Begruendung -- auch wenn letztere im Alltag der "
    "echte Grund waere."
)


_TRIAGE_PROMPT = (
    "Dies ist ein Screenshot einer Fahrschul-Lern-App.\n"
    "Klassifiziere den gezeigten Screen:\n"
    "- 'foto': Bildfrage mit statischem Foto und Antwort-Checkboxen\n"
    "- 'text': reine Textfrage mit Antwort-Checkboxen (kein grosses Bild)\n"
    "- 'film': Videofrage, erkennbar an Video-Bereich mit Play-Symbol oder "
    "'Film starten'-Button; Video kann noch nicht gestartet, spielend oder beendet sein\n"
    "- 'input': Frage mit einem Texteingabefeld (z.B. ein weisser/hellgrauer "
    "'Antwort'-Kasten neben einer Einheit wie 'km/h'), KEINE Checkboxen\n"
    "- 'other': anderer Screen (Menu, Loesungsanzeige, Erklaerung, Startseite)\n\n"
    "Falls question_type='film':\n"
    "- video_playing=true wenn das Video aktuell abspielt\n"
    "- video_playing=false wenn nicht, und lokalisiere dann den START-BUTTON "
    "als video_start_button (BoundingBox 0-1000). WICHTIG: bevorzuge den klar "
    "beschrifteten Button mit Text wie 'Film starten' oder 'Start' (meist unten "
    "im Fenster als vollwertiger Button), NICHT das kleine Play-Dreieck-Overlay "
    "mitten im Video. Nur falls kein beschrifteter Text-Button sichtbar ist, "
    "nimm das zentrale Play-Symbol.\n"
    "Sonst video_playing=false und video_start_button=null."
)


_FOTO_PROMPT = (
    "Screenshot einer offiziellen deutschen Fahrschul-Theoriefrage.\n"
    "Beantworte die Frage mit deiner Fahrlehrer-Methodik (Schritt 1-4). Fuehre die "
    "schrittweise Analyse in explanation aus, bevor du correct_solutions festlegst.\n\n"
    "Lokalisiere fuer jede sichtbare Antwortoption die CHECKBOX -- das kleine quadratische "
    "Klick-Icon LINKS neben dem Antworttext, nicht den Text selbst. Gib die Bounding-Box "
    "als ymin, xmin, ymax, xmax im normalisierten Bereich 0-1000 zurueck."
)


_INPUT_PROMPT = (
    "Screenshot einer offiziellen deutschen Fahrschul-Theoriefrage mit einem "
    "Eingabefeld fuer eine typisierte Antwort (meist eine Zahl, z.B. "
    "Geschwindigkeit in km/h, Meter, Sekunden).\n"
    "Beantworte die Frage mit deiner Fahrlehrer-Methodik. Fuehre die schrittweise "
    "Analyse in explanation aus, bevor du answer_value festlegst. Achte besonders "
    "auf die gefragte Einheit und etwaige Zahlen in der Fragestellung.\n\n"
    "Lokalisiere das Eingabefeld (der Antwortkasten mit Placeholder 'Antwort' "
    "oder aehnlich) und gib dessen Bounding-Box im normalisierten Bereich 0-1000 "
    "(ymin, xmin, ymax, xmax) zurueck."
)


_INPUT_FILM_PROMPT = (
    "Videomaterial einer offiziellen deutschen Fahrschul-Filmfrage plus ein "
    "Screenshot nach dem Ende des Videos. Die Frage erwartet eine typisierte "
    "Antwort in einem Eingabefeld.\n"
    "Beantworte die Frage mit deiner Fahrlehrer-Methodik. Beruecksichtige die "
    "Verkehrssituation aus dem Video (Geschwindigkeiten, Abstaende, Handlungen) "
    "und die genaue Formulierung der Frage. Fuehre die schrittweise Analyse in "
    "explanation aus, bevor du answer_value festlegst.\n\n"
    "Lokalisiere das Eingabefeld im Screenshot und gib dessen Bounding-Box "
    "(0-1000 normalisiert) zurueck."
)


_FILM_PROMPT = (
    "Videomaterial einer offiziellen deutschen Fahrschul-Filmfrage plus ein "
    "Screenshot nach dem Ende des Videos.\n"
    "Beantworte die am Ende gestellte Frage mit deiner Fahrlehrer-Methodik. "
    "Analysiere die Verkehrssituation aus dem Video (Fahrzeugpositionen, "
    "Geschwindigkeiten, Verkehrsschilder, Gefahrenquellen) und pruefe jede "
    "Antwortoption einzeln. Fuehre die schrittweise Analyse in explanation aus, "
    "bevor du correct_solutions festlegst.\n\n"
    "Lokalisiere fuer jede sichtbare Antwortoption im Screenshot die CHECKBOX "
    "(das kleine quadratische Klick-Icon LINKS neben dem Antworttext) und gib "
    "die Bounding-Box als ymin, xmin, ymax, xmax im normalisierten Bereich "
    "0-1000 zurueck."
)


class BundleEndScreen(BaseModel):
    is_bundle_end: bool = Field(
        description=(
            "True wenn der Screen eine Zusammenfassung eines abgeschlossenen "
            "Fragenbuendels ist (vollflaechiger farbiger Hintergrund mit Score-Text "
            "und einem grossen Fortsetzen-Button am unteren Rand), sonst False."
        )
    )
    score_text: str | None = Field(
        default=None,
        description="Falls sichtbar: der Score-Text wie '22/30 Fragen richtig beantwortet'."
    )
    continue_button: BoundingBox | None = Field(
        default=None,
        description=(
            "Bounding-Box (0-1000) des HAUPT-Fortsetzungs-Buttons am unteren Rand. "
            "Button-Text variiert ('Falsche Fragen ueben', 'Lernweg fortsetzen', "
            "'Weiter ueben' usw.) -- waehle den groessten, prominentesten "
            "Call-to-Action unten. NICHT das kleine X-Schliessen-Icon oben rechts."
        )
    )
    reasoning: str = Field(description="Kurze Begruendung.")


class HomeScreenDetection(BaseModel):
    is_home_screen: bool = Field(
        description=(
            "True wenn der Screen der Startbildschirm/Hauptbildschirm der Fahrschul-App ist."
        )
    )
    top_active_card: BoundingBox | None = Field(
        default=None,
        description=(
            "Bounding Box (0-1000) der TOPMOST AKTIVEN Lerneinheit-Karte: die "
            "oberste Lerneinheit, die NICHT gesperrt ist (kein Schloss-Symbol). "
            "Null wenn kein Home-Screen."
        )
    )
    card_label: str = Field(
        default="",
        description="Sichtbarer Titel der Karte (z.B. 'Lerneinheit 5/9'), fuer Logging."
    )
    reasoning: str = Field(description="Kurze Begruendung.")


class ExtractedMusterloesung(BaseModel):
    """Correct answers extracted from the post-submit feedback screen."""

    question_type: Literal["foto", "text", "input"] = Field(
        description="foto/text = MC mit/ohne Bild; input = Texteingabe."
    )
    we_were_correct: bool = Field(
        description="True wenn 'Deine Antwort ist richtig' (gruener Text) sichtbar."
    )
    correct_indices: list[int] = Field(
        default_factory=list,
        description=(
            "Nur bei MC: 1-basierte Indizes der in der MUSTERLOESUNGS-Spalte (links, "
            "grau) angehakten Optionen, aufsteigend sortiert."
        )
    )
    correct_value: str | None = Field(
        default=None,
        description=(
            "Nur bei input: der im Musterloesungs-Eingabefeld stehende EXAKTE Wert "
            "ohne Einheit."
        )
    )


def locate_button(
    client: GeminiClient, capture: Capture, button_text: str
) -> ButtonLocation:
    prompt = (
        f"Lokalisiere im Screenshot einen Button mit dem Text \"{button_text}\". "
        "Falls gefunden, gib die Bounding-Box als BoundingBox im normalisierten "
        "Bereich 0-1000 (ymin, xmin, ymax, xmax) zurueck und found=true. "
        "Falls kein Button mit genau diesem Text sichtbar ist, found=false, box=null."
    )
    return client.ask(
        prompt=prompt,
        image=capture.image,
        schema=ButtonLocation,
        temperature=0.1,
    )


def detect_home_screen(
    client: GeminiClient, capture: Capture
) -> HomeScreenDetection:
    prompt = (
        "Dies ist ein Screenshot einer Fahrschul-Lern-App.\n"
        "Pruefe strukturell ob der Screen der Startbildschirm (Home / Lernweg) ist. "
        "Erkennungsmerkmale:\n"
        "- 'PREMIUM' Label oben links\n"
        "- Prozent-Anzeige (z.B. '20%') in rotem Kreis oben Mitte\n"
        "- 'Theorie' und 'Praxis' Sektions-Headlines\n"
        "- Mehrere grosse rechteckige Karten uebereinander: 'Lerneinheit X/9' "
        "(Verkehrszeichen-Bild, Thema-Info), 'E-Book', 'DRIVERS CAM'\n"
        "- Manche Karten sind GESPERRT (dunkel/grau mit Schloss-Symbol unten rechts)\n"
        "- Untere Tab-Navigation: Lernweg / Ueben / To-do / Mehr\n\n"
        "Falls ja: is_home_screen=true, und lokalisiere die TOPMOST AKTIVE Karte -- "
        "das ist die oberste 'Lerneinheit'-Karte, die NICHT gesperrt ist (kein "
        "Schloss-Symbol sichtbar). Gib deren Bounding-Box als top_active_card "
        "(0-1000 normalisiert) zurueck. Setze card_label auf den sichtbaren Titel "
        "der Karte (z.B. 'Lerneinheit 5/9').\n"
        "Falls nicht: is_home_screen=false, top_active_card=null."
    )
    return client.ask(
        prompt=prompt,
        image=capture.image,
        schema=HomeScreenDetection,
        temperature=0.0,
    )


def detect_bundle_end(
    client: GeminiClient, capture: Capture
) -> BundleEndScreen:
    prompt = (
        "Dies ist ein Screenshot einer Fahrschul-Lern-App.\n"
        "Pruefe strukturell, ob der Screen eine ZUSAMMENFASSUNG eines abgeschlossenen "
        "Fragenbuendels ist. Erkennungsmerkmale:\n"
        "- Vollflaechige farbige Hintergrundflaeche (blau/lila/rot/gruen), "
        "NICHT die normale weisse Frageansicht\n"
        "- Ein Score-Text wie 'X/Y Fragen richtig beantwortet'\n"
        "- Ein grosser Haupt-Button am unteren Rand zum Fortsetzen oder Wiederholen\n"
        "- Oft ein Emoji oder Grafik in der Mitte\n"
        "- Ein kleines 'X'-Schliessen-Icon oben rechts (NICHT der gesuchte Button)\n\n"
        "Wenn ja: is_bundle_end=true, score_text, und continue_button als "
        "Bounding-Box des grossen unteren Buttons (0-1000 normalisiert).\n"
        "Wenn nicht: is_bundle_end=false und continue_button=null."
    )
    return client.ask(
        prompt=prompt,
        image=capture.image,
        schema=BundleEndScreen,
        temperature=0.0,
    )


def extract_musterloesung(
    client: GeminiClient, capture: Capture
) -> ExtractedMusterloesung:
    prompt = (
        "Dies ist ein Feedback-Screenshot nach dem Einreichen einer Fahrschul-"
        "Theoriefrage. Die MUSTERLOESUNG ist auf dem Screen sichtbar:\n"
        "- Bei MC-Fragen: in der LINKEN GRAUEN Spalte neben den (rechten, dunkleren) "
        "eigenen Antworten\n"
        "- Bei Input-Fragen: als zweites Eingabefeld unterhalb der eigenen Eingabe "
        "(mit grauer Schrift)\n\n"
        "Extrahiere:\n"
        "- question_type: foto/text fuer MC-Fragen (foto bei grossem Bild oder Video-"
        "Thumbnail, text bei reinem Text), input bei Eingabefeld-Fragen\n"
        "- we_were_correct: true wenn der gruene Text 'Deine Antwort ist richtig' "
        "sichtbar ist\n"
        "- Bei MC: correct_indices = 1-basierte Indizes der in der MUSTERLOESUNGS-"
        "Spalte angehakten Optionen, aufsteigend sortiert, von oben gezaehlt\n"
        "- Bei input: correct_value = der im Musterloesungs-Feld stehende EXAKTE Wert "
        "(ohne Einheit)"
    )
    return client.ask(
        prompt=prompt,
        image=capture.image,
        schema=ExtractedMusterloesung,
        temperature=0.0,
    )


def triage_screen(client: GeminiClient, capture: Capture) -> ScreenTriage:
    return client.ask(
        prompt=_TRIAGE_PROMPT,
        image=capture.image,
        schema=ScreenTriage,
        temperature=0.1,
    )


def solve_question(client: GeminiClient, capture: Capture) -> FahrschulAntwort:
    return client.ask(
        prompt=_FOTO_PROMPT,
        image=capture.image,
        schema=FahrschulAntwort,
        system_instruction=_INSTRUCTOR_SYSTEM,
        temperature=0.0,
    )


def solve_input_question(
    client: GeminiClient, capture: Capture
) -> FahrschulInputAntwort:
    return client.ask(
        prompt=_INPUT_PROMPT,
        image=capture.image,
        schema=FahrschulInputAntwort,
        system_instruction=_INSTRUCTOR_SYSTEM,
        temperature=0.0,
    )


def solve_input_film_question(
    client: GeminiClient,
    final_frame: Capture,
    video_path: Path,
) -> FahrschulInputAntwort:
    """Two-pass solver for film input questions.

    Pass 1: video + final frame produces the answer value and explanation.
    Pass 2: image-only locates the input field (coord accuracy is noticeably
    better without the video context sharing the model's attention).
    """
    semantic = client.ask_with_video(
        prompt=_INPUT_FILM_PROMPT,
        video_path=video_path,
        image=final_frame.image,
        schema=FahrschulInputAntwort,
        system_instruction=_INSTRUCTOR_SYSTEM,
        temperature=0.0,
    )
    layout = solve_input_question(client, final_frame)
    return FahrschulInputAntwort(
        explanation=semantic.explanation,
        answer_value=semantic.answer_value,
        input_field=layout.input_field,
    )


def solve_film_question(
    client: GeminiClient,
    final_frame: Capture,
    video_path: Path,
) -> FahrschulAntwort:
    """Two-pass solver for film MC questions.

    Pass 1 analyses the video + final frame to produce the semantic answer
    (correct_solutions + explanation). Pass 2 is image-only coordinate
    grounding, which flash-lite does noticeably better without video context.
    Results are merged.
    """
    semantic = client.ask_with_video(
        prompt=_FILM_PROMPT,
        video_path=video_path,
        image=final_frame.image,
        schema=FahrschulAntwort,
        system_instruction=_INSTRUCTOR_SYSTEM,
        temperature=0.0,
    )
    layout = solve_question(client, final_frame)
    return FahrschulAntwort(
        explanation=semantic.explanation,
        correct_solutions=semantic.correct_solutions,
        answer_boxes=layout.answer_boxes,
    )
