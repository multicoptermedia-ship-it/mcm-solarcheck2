# Optionale externe RGB-Kamerakalibrierung – Anforderungen

Stand: 30.10.2026. **Optionaler geprüfter Pfad, keine Voraussetzung für die normale RGB-Orthofoto-Entwicklung.** Der Nutzer hat die kurze vorherige Beschaffungsentscheidung ausdrücklich ersetzt: Die vorhandene Mehrbildpipeline wird ohne externe Kalibrierungsdatei mit ungeprüfter EXIF-Kamerainitialisierung und gemeinsamer Brennweitenoptimierung weiterentwickelt. Diese Datei bewahrt die Anforderungen für den Fall, dass später zusätzlich ein unabhängiger externer Nachweis beschafft wird. Maßgeblich bleibt [AKTUELLER_ENTWICKLUNGSSTAND.md](AKTUELLER_ENTWICKLUNGSSTAND.md).

## Welche Kamera und Bilder?

Eine repräsentative, tatsächlich heruntergeladene Originaldatei nennt `Make=DJI`, `Model=M3T` und enthält das RGB-Hauptbild mit **4000×3000 Pixeln** im JPEG-/MPO-Container. Gesucht ist die Kalibrierung der **RGB-Kamera für diese Originalausgabe**, keine Thermal-/Temperaturkalibrierung.

Die Kalibrierung muss zum konkreten Kameragerät, Objektiv und verwendeten Aufnahme-/Ausgabemodus passen. Auflösung, Seitenverhältnis, Ausschnitt/Binning, Zoom-/Fokuseinstellung und bereits angewandte kamerainterne Entzerrung müssen geklärt sein. Gleiche Pixelabmessungen allein beweisen diese Übereinstimmung nicht. Seriennummern und gerätebezogene Unterlagen nur privat austauschen, nicht in GitHub ablegen.

## Benötigte Werte und Nachweise

| Bestandteil | Benötigte Angaben |
| --- | --- |
| Intrinsics | `fx`, `fy`, `cx`, `cy` in **Pixeln des 4000×3000-Originalrasters**, samt Koordinatenkonvention. Das bestehende Modell verwendet eine Kameramatrix ohne Scherung. |
| Verzeichnung | Koeffizienten mit eindeutigem Modell, Reihenfolge und Konvention. Der bestehende OpenCV-Pfad unterstützt 4/5/8/12/14 Werte. |
| Bildverarbeitung | Angabe, ob die ausgegebenen JPEGs bereits entzerrt sind und für welchen konkreten Bildzustand die Parameter gelten. Im verifizierten externen Pfad Nullkoeffizienten nicht ohne Nachweis als geprüfte Werte einsetzen. Der automatische Kandidatenpfad nutzt Nullverzeichnung ausdrücklich nur als ungeprüfte Startannahme. |
| Zuordnung | Welches Gerät/Objektiv und welcher Aufnahme-/Ausgabemodus wurden kalibriert? Für welche Auflösung gelten die Werte? |
| Provenienz | Aussteller, Datum, Methode und nachvollziehbare Kalibrierungsunterlagen. |
| Validierung | Dokumentierte Fehler-/Genauigkeitsbewertung und möglichst separate Prüfbilder oder ein unabhängiger Prüfbericht. Ein kleines Anpassungsresiduum der verwendeten Kalibrierbilder allein ist kein unabhängiger Nachweis. |

Allgemeine Datenblatt-Brennweite, Kleinbildäquivalent, GNSS-/RTK-Daten, Gimbalwinkel oder die bloße Übereinstimmung der RGB-/Thermal-Bildnummern ersetzen diesen Nachweis nicht. Eine erfolgreiche vollständige Rekonstruktion des aktuellen Bestands ist noch nicht nachgewiesen. ORB-Läufe ohne externe Datei scheiterten am Track-/PnP-Stützungsproblem, nicht an einem externen Dateigate. Der gleiche Dienst/CLI nutzt nun SIFT als Standard (`--feature-method sift`, optional `orb`); alle 295 Merkmalsbilder/1.958 Matchings/1.850 Essential-Prüfungen sind berechnet und privat gesichert, 546 geometrische Links verbinden alle Kameras. Der globale Dreisicht-Start korrigiert die erzwungene erste Startkamera; der echte Ganzflugpfad scheitert dennoch bei 3/295 Posen und 176 Landmarken an unzureichender persistenter 3D-Track-/PnP-Stützung, BA nicht erreicht. Kein externes Dateigate ist die Ursache. Nächste Arbeit ist Mehrbild-Track-/RTK-Initialisierung in der bestehenden Engine, keine erneute Beschaffungsaufgabe. Nach SIFT sind 4.063 lokale Tests bestanden (15 Skips/14 Warnungen), `pip check` und Offline-Qt-Smoke erfolgreich. Code-HEAD `888626e219e9d5a0d2ed932006a962816debfeea` hat erfolgreiche Python-/Windows-CI unter 3.11/3.12; Links und Grenzen im [Startdokument](SOLARCHECK2_START.md). Für die folgende Startkorrektur sind 12 Kameranetztests und 45 ausgewählte Pipeline-/BA-/Netztests bestanden; vollständige Suite/CI am exakten aktuellen HEAD prüfen.

## Praktischer Beschaffungsweg

1. Beim Lieferanten beziehungsweise [DJI Enterprise Support](https://www.dji.com/support/product/mavic-3-enterprise) gezielt nach einer zu diesem Gerät und JPEG-Ausgabemodus passenden RGB-Kalibrierung fragen. Ein weiterer offizieller Einstieg ist [DJI Service](https://www.dji.com/support/repair).
2. Es ist bisher nicht belegt, dass DJI solche gerätespezifischen Dateien öffentlich oder auf Anfrage bereitstellt. Die [offizielle Produkt-Downloadseite](https://enterprise.dji.com/mavic-3-enterprise/downloads) enthält keine hier nachgewiesene passende RGB-Intrinsics-Datei. ThermalSDK ist kein Ersatz für RGB-Kalibrierung.
3. Falls kein passender Hersteller-/Lieferantennachweis verfügbar ist: ein fachkundiges Kamerakalibrierungs-/Photogrammetrielabor mit genau diesem Gerät und passendem Bildmodus beauftragen. Vorher klären, ob dessen Kalibrierung auf die bereits vorhandenen Aufnahmen anwendbar ist. Kalibrierbilder dürfen getrennt aufgenommen werden; Originalflugaufnahmen bleiben unverändert.

Es wurden keine Anbieter kontaktiert, Aufträge erteilt oder Kosten ausgelöst.

## Kurzer Anfragetext

> Für DJI M3T-RGB-Originalaufnahmen mit 4000×3000 Pixeln benötigen wir eine gerätebezogene Kamerakalibrierung. Können Sie passende Intrinsics (`fx`, `fy`, `cx`, `cy` in Originalpixeln), Verzeichnungsparameter mit Modell/Konvention sowie einen Kalibrierungs- und Validierungsnachweis bereitstellen? Bitte bestätigen Sie den unterstützten Aufnahme-/Zoom-/Fokusmodus und ob die Parameter für die bereits ausgegebenen, gegebenenfalls kameraintern entzerrten JPEG-/MPO-Bilder gelten. Wenn keine solche Datei verfügbar ist: Welche Kalibrierungsleistung und Unterlagen können Sie für dieses konkrete Gerät anbieten?

## Übernahme in SolarCheck 2

Herstellerdatei, OpenCV-YAML/XML oder Bericht zunächst privat zur Prüfung bereitstellen. Niemand muss ungeprüft ein SolarCheck-JSON ausfüllen. Passt das Modell, werden die echten Werte nachvollziehbar in den vorhandenen Vertrag `mcm.rgb_camera_intrinsics_external.v1` übertragen. Der optionale CLI-Parameter `--calibration-json` liest dieses JSON; ohne ihn nutzt der bestehende Ganzflugpfad die automatische ungeprüfte Kamerainitialisierung. Ein allgemeiner Hersteller-Konverter ist noch nicht vorhanden.

`lens_calibration_verified` darf erst bei tatsächlich geprüftem passenden Nachweis gesetzt werden. Keine Zahlen aus Beispieltests oder nominalen Millimeter-Brennweiten übernehmen und kein Prüfmerkmal zum Umgehen des Eingangsschutzes setzen. Dies ist ausschließlich die Bedingung für den **extern verifizierten** Pfad. Die lokal vorbereitete vollständige RGB-Serie darf im automatischen SfM-/RTK-BA-Kandidatenpfad bereits ohne diese Datei verarbeitet werden. Vermessungs-, Dach-/Orthofoto-, PV- und Thermal-Freigaben bleiben separate Gates.
