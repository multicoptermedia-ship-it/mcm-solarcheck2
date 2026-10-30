# AKTUELLER ENTWICKLUNGSSTAND – MCM-SolarCheck 2
**Stand: 30.10.2026. Verbindlicher Erstlesepunkt jedes neuen Entwicklungschats. Neueste Nutzerentscheidung, Code und CI haben Vorrang.**

## Ziel und Arbeitsgrenzen
Vollständige DJI-M3T-Serie → eigene Mehrbild-Photogrammetrie (SfM) mit GNSS/RTK → Kameraposen/Dachflächen → vorläufiges RGB-Orthofoto → PV-Modulprüfung → Thermalregistrierung/Befunde.
**ODM / WebODM / ODX NICHT integrieren, aufrufen, kopieren oder paketieren.** Offline/Windows 10, Datenschutz und Lizenzprüfungen gelten. GNSS-EXIF ist kein unabhängiger 3D-/Zentimeternachweis; unabhängige Kontrollen bleiben für geprüfte absolute Genauigkeit erforderlich.

**Alle Änderungen ausschließlich in `multicoptermedia-ship-it/mcm-solarcheck2`, Branch `development/rgb-photogrammetry`.** `MCM-SolarCheck` samt lokalem Checkout nur lesen: keine Datei-/Git-Metadatenänderungen, Commits, Pushes, PR-Änderungen oder Merges. **Kein Merge/Push nach `main` ohne ausdrückliche Nutzerfreigabe.** [AGENTS.md](../AGENTS.md), [SOLARCHECK2_START.md](SOLARCHECK2_START.md).

## RGB ohne obligatorische externe Kalibrierungsdatei
Der Nutzer hat die kurze vorherige Beschaffungsentscheidung ausdrücklich ersetzt: **Die normale RGB-Orthofoto-Entwicklung wird nicht durch eine fehlende externe Kamerakalibrierungsdatei blockiert.** Die bestehende Mehrbildpipeline mit allen 295 verfügbaren Originalen fortsetzen; keine erneute Nachfrage nach bereits verneinter Datei.

`rgb_camera_initialization.py`: EXIF-Kleinbildäquivalent + Originalabmessungen liefern Startintrinsics; Bildmitte, quadratische Pixel und Nullverzeichnung sind **ungeprüfte Annahmen**. Keine Ableitung allein aus 4,4 mm. Schur-/Fenster-/RTK-BA optimiert eine gemeinsame Brennweitenskalierung, keine vollständige Selbstkalibrierung von Hauptpunkt/Seitenverhältnis/Verzeichnung. Optionaler geprüfter externer Pfad: `--calibration-json`, [Anforderungen](RGB_CAMERA_CALIBRATION_REQUIREMENTS.md).

`lens_calibration_verified`, Dach-/Orthofoto-/Vermessungs-/PV-/Thermalfreigaben bleiben im automatischen Kandidatenweg falsch. Keine erfundenen `VerifiedCameraGeometry`-Objekte. Vorläufige Rekonstruktion und fachliche Freigabe getrennt ausweisen.

## Code und geprüfte Ergebnisse
**Bisher nachgewiesener Code-HEAD `888626e219e9d5a0d2ed932006a962816debfeea`:** ausschließlich im Ziel-Entwicklungsbranch gepusht; [Python-CI](https://github.com/multicoptermedia-ship-it/mcm-solarcheck2/actions/runs/41092574805) und [Windows-CI](https://github.com/multicoptermedia-ship-it/mcm-solarcheck2/actions/runs/41092574880) unter 3.11/3.12 erfolgreich. Damalige Gesamtsuite nach SIFT: **4.063 bestanden, 15 Skips, 14 Warnungen (130,30 s)**; `pip check`/Offline-Qt-Smoke erfolgreich. Danach neue Startkorrektur: 12 Kameranetztests einschließlich zweier Sichtbarkeits-/Bezugssystemtests sowie 45 ausgewählte Pipeline-/BA-/Netztests bestanden. Abschließende lokale Gesamtsuite nach der Startkorrektur: **4.065 bestanden, 15 Skips, 14 Warnungen (133,53 s)**. **CI am exakten aktuellen HEAD prüfen**; ältere grüne Läufe gelten für ihren jeweiligen Code. Keine Windows-10-Hardware-/Orthofoto-Fachabnahme.

Referenz-`main` unverändert `5f2de8fa`, Basis PR #9 `6b72e8f3`; Dienste/Gates aus grünem `f616d9dc` gezielt wiederverwendet, kein Vollmerge. PR #7, PR #8 und PR #9/volle SHAs/CI-Belege: [SOLARCHECK2_START.md](SOLARCHECK2_START.md).

## Vorhandenen Gesamtflugweg weiterführen
**DEV-RGB-ORTHO-ENGINE-001:** vollständige private ZIP/ZIP-kompatible `.tip`-Serie → SIFT (Standard; ORB optional) → globales SfM-Tracknetz → überlappende Fenster-BA → RTK-gewichtete Fenster-BA. `rgb_zip_whole_flight_reconstruction.py` und `rgb_zip_whole_flight_cli.py` verbinden die Dienste; keine zusätzliche Photogrammetrielaufzeit oder Abhängigkeit.
- Kameragraph: alle Bilder verbunden, mindestens zwei räumlich nicht benachbarte Essential-RANSAC-/Pose-akzeptierte Links mit mindestens drei Kameras; Deskriptortreffer allein reichen nicht. Trackkonflikt-, Kameraresiduen- und Parallaxengates bleiben bestehen.
- GNSS/RTK-Priors mit konservativen Unsicherheiten; 20 % der Kameras bei Ausrichtung/Optimierung zurückgehalten. Interne RTK-Hold-outs sind keine unabhängigen GCPs. Fenster-Schur ist keine simultane globale BA aller Kameras.
- Vorhandener Ebenenfitter → begrenzte Delaunay-Stützung ohne Lückenfüllung → Strahl-/Ebenenverdeckung → gemeinsame XY-Rückprojektion. `--candidate-output-dir` schreibt private PNG-/NoData-Kandidaten erst nach vollständigen Geometriegates; Auflösungsoptionen siehe CLI-Hilfe. Ausgaben bleiben ungeprüft.

## Reale Daten und tatsächlicher Laufstand
Drive einschließlich Originaldownload erreichbar: **295 RGB, 287 Thermal, 287 eindeutige Bildnummernpaare und acht RGB ohne gelisteten Thermalpartner.** Alle 295 Originale außerhalb von Git: 2.214.162.432 Bytes, private SHA256-Nachweise, byteidentischer CRC-geprüfter ZIP-Eingang. Historische 160 sind keine Sollgrenze; Flug-/Dachabdeckung ungeprüft. URLs, Fotos, GPS/Rekonstruktionsartefakte nicht in GitHub speichern.

ORB-Läufe scheiterten an Track-/PnP-Stützung; Details im Startdokument. SIFT ist Standard im selben Dienst/CLI (`--feature-method sift`, Legacy `orb`), vorhandenes OpenCV.

**SIFT-Realstand:** Alle 295 Merkmalsbilder, 1.958 Paar-Matchings und 1.850 Essential-Prüfungen berechnet/privat gesichert; 546 akzeptierte Links verbinden alle Bilder. Der Kamera-0-Zwangsstart ist korrigiert: begrenzte globale Dreisicht-Initialisierung (höchstens 64 Startpaare), anschließend Bezugssystem der ersten Eingabekamera. Echter Ganzflugpfad jetzt **3/295 Posen, 176 Landmarken**, dann global_sfm/PnP-FAIL wegen fehlender 3D-Stützung; BA nicht erreicht. Private Diagnosen erreichen höchstens 11/295 Kameras mit vorhandener BA bzw. 14/295 mit nicht integriertem All-View-DLT-Prototyp. Kein spekulativer BA-/DLT-Fix übernommen, keine erfolgreiche Ganzflug-/Dachflächen-/Orthofotoausgabe. Details im Startdokument.

## Nächster Schritt und Übergabe
Persistente Mehrbild-Tracks und RTK-gestützte Initialisierung in der vorhandenen Engine verbessern, danach vollständige Fluggeometrie und RGB-Orthorektifizierung nachweisen. Berechnete Merkmale/Matches/Essential-Zwischenstände wiederverwenden; keine Kalibrierungsbeschaffung, zusätzliche Gates oder partielles 2D-Stitching. **Keine erneuten ZIP-/EXIF-/MPO-/ORB-Inventuren**, Paar-, Surveyformular-, ODM- oder KI-Nebenarbeit. Originale/produktive SQLite unverändert; Windows-Nutzertests nur an separater Datenbankkopie. Den laut Nutzer bereits im Referenzprojekt untersuchten mittigen Thermal-Temperaturpunkt und vorhandene Thermalparser später wiederverwenden; zuerst RGB. Bildnummernpaare allein sind keine Sensorregistrierung.

Cloud-Konfigurationsentwurf mit aktuellem Code-HEAD/Startanweisungen gespeichert; bestehendes unbeschränktes Netzwerk beibehalten. Veröffentlichung und Wiederherstellung in neuer Aufgabe bleiben unbestätigt, blockieren aber nicht die Arbeit hier.

Nach jedem Block zuerst Checkpoint (maximal 8.500 Zeichen), dann [DEVELOPMENT_STEP_REGISTER.md](DEVELOPMENT_STEP_REGISTER.md), [DEVELOPMENT_AGREEMENT.md](DEVELOPMENT_AGREEMENT.md), [CHAT_HANDOFF.md](CHAT_HANDOFF.md) und [PV_ORTHOMOSAIC_FIRST_ROADMAP.md](PV_ORTHOMOSAIC_FIRST_ROADMAP.md) synchronisieren. Abschluss-HEAD/CI und reale Grenzen ergänzen.
