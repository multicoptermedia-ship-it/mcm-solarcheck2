# SolarCheck 2 – Entwicklung im eigenen Zielrepository

Stand: 30.10.2026. Zuerst [AKTUELLER_ENTWICKLUNGSSTAND.md](AKTUELLER_ENTWICKLUNGSSTAND.md) lesen. Diese Datei ergänzt den kurzen Checkpoint um Übernahme, nachgewiesene CI, Cloud-Start und den aktuellen privaten Gesamtflugweg.

## Repository und Arbeitsgrenzen

- Ausschließliches Entwicklungsziel: `multicoptermedia-ship-it/mcm-solarcheck2`, lokal `/workspace/mcm-solarcheck2`.
- Entwicklungsbranch: `development/rgb-photogrammetry`; kein Merge/Push nach `main` ohne ausdrückliche Nutzerfreigabe.
- Übernommene Codebasis: `6b72e8f373c3eb8f61615916705699b048616531` aus dem Referenzbranch `feature/orthofoto-license-neutral-rgb-first-20261028`, einschließlich vorhandener Historie und fertig entwickelter Bausteine.
- `multicoptermedia-ship-it/MCM-SolarCheck` einschließlich `/workspace/MCM-SolarCheck` ausschließlich lesen. Keine Datei-/Git-Metadatenänderungen, Commits, Pushes, PR-Änderungen, Merges oder dort gestarteten CI-Läufe. Lokaler Referenz-HEAD unverändert `5f2de8fa47c7ece5b721ca46b9a29c8c5b194059`.
- Originale, private Rekonstruktionsartefakte und produktive SQLite unverändert lassen. Bilder, GPS, private URLs und Kameraunterlagen nicht in GitHub speichern. Keine zusätzlichen Worktrees oder wiederholte Grundinventuren. [AGENTS.md](../AGENTS.md).

## Aktuelle Nutzerentscheidung und vorhandene Eingaben

**Der normale RGB-Orthofoto-Entwicklungsweg benötigt keine externe Kalibrierungsdatei als Voraussetzung.** Der Nutzer hat die kurze vorherige Entscheidung, zunächst eine solche Datei zu beschaffen, ausdrücklich ersetzt. Die bestehende eigene Mehrbildpipeline wird weiterentwickelt und mit der vollständigen verfügbaren RGB-Serie ausgeführt.

`rgb_camera_initialization.py` nutzt tatsächliches EXIF-Kleinbildäquivalent und Originalabmessungen für eine nominale gemeinsame Kamera. Hauptpunkt in Bildmitte, quadratische Pixel und Nullverzeichnung sind ungeprüfte Startannahmen. Die gemeinsame Brennweite kann in der vorhandenen Schur-/Fenster-/RTK-BA aus Mehrbildresiduen mitoptimiert werden. Dies ist keine vollständige Selbstkalibrierung von Hauptpunkt, Seitenverhältnis und Verzeichnung. `lens_calibration_verified` sowie Dach-/Orthofoto-/PV-/Thermalfreigaben bleiben im automatischen Kandidatenweg falsch. Keine Zahlen allein aus 4,4 mm ableiten und keine verifizierten Geometrieobjekte erfinden.

Der bestehende streng geprüfte externe Pfad bleibt optional: `--calibration-json` akzeptiert weiterhin `mcm.rgb_camera_intrinsics_external.v1` mit passenden `fx/fy/cx/cy`, OpenCV-Verzeichnung und unabhängig geprüfter Provenienz. [RGB_CAMERA_CALIBRATION_REQUIREMENTS.md](RGB_CAMERA_CALIBRATION_REQUIREMENTS.md) dokumentiert diesen optionalen Zusatzweg; keine erneute Beschaffungsnachfrage als Blocker.

Drive-Ordner und Originaldownload sind erreichbar. Die vollständige Ansicht enthält **295 RGB, 287 Thermal, 287 eindeutige Bildnummernpaare und acht RGB ohne gelisteten Thermalpartner**. Die ersten 50 Einträge der Hauptansicht waren nicht vollständig. Alle 295 RGB-Originale sind außerhalb beider Git-Checkouts unverändert geladen (2.214.162.432 Bytes), private SHA256-Nachweise vorhanden, byteidentischer ZIP-Eingang mit geprüftem CRC fertig. Die historischen sechs Archive mit 160 RGB bleiben dokumentierte frühere Eingabe; den neuen vollständigen Bestand nicht künstlich auf 160 reduzieren. Physische Flug-/Dachabdeckung noch ungeprüft. Ein zugehöriges Originalpaar wurde strukturell als 4000×3000 RGB / 640×512 Thermal geprüft; keine neue Temperatur-/Registrierungsanalyse.

Der Nutzer bestätigt die bereits untersuchte mittige Temperaturinformation der Thermalbilder im Referenzprojekt. Vorhandene Erkenntnisse und Thermalparser später wiederverwenden; zuerst RGB-Rekonstruktion und Orthofoto, daraus PV-Modulprüfung, anschließend Thermal. Gleiche Bildnummer ordnet Aufnahmen zu und ist allein kein geometrischer Registrierungsnachweis.

## Bereits verifizierte Ergebnisse

Referenzprüfungen vom 29./30.10.2026 wurden nur gelesen, nicht neu im Referenzrepository gestartet:

| Referenz-PR | Vollständiger HEAD | Python-CI | Windows-CI |
| --- | --- | --- | --- |
| #7 | `14842d98ed00c4c6f9416dc3157149ce4d0ac721` | [erfolgreich](https://github.com/multicoptermedia-ship-it/MCM-SolarCheck/actions/runs/40561014910) | [erfolgreich](https://github.com/multicoptermedia-ship-it/MCM-SolarCheck/actions/runs/40561014896) |
| #8 | `2f1ac0e967415308b7ab97e78e83d5b32d99f7c6` | [erfolgreich](https://github.com/multicoptermedia-ship-it/MCM-SolarCheck/actions/runs/40562888287) | [erfolgreich](https://github.com/multicoptermedia-ship-it/MCM-SolarCheck/actions/runs/40562889517) |
| #9, neuerer gelesener Stand | `f616d9dcd4496c054acdcf869ec3ff7d9ba82627` | [erfolgreich](https://github.com/multicoptermedia-ship-it/MCM-SolarCheck/actions/runs/40944490236) | [erfolgreich](https://github.com/multicoptermedia-ship-it/MCM-SolarCheck/actions/runs/40944490295) |

Die ursprüngliche Übernahmebasis bleibt `6b72e8f373c3eb8f61615916705699b048616531`; der später gelesene `f616d9dc` ist kein vollständiger Merge der späteren Referenzhistorie. Dessen vorhandene `rgb_private_pitched_plane_candidates.py`, `rgb_private_pitched_plane_orthorectification.py`, `rgb_private_multi_plane_preview.py` samt Originaltests sowie Track-/Kameraresiduen-/Parallaxengates werden gezielt im Ziel wiederverwendet. Eigener doppelter Ebenen-RANSAC wurde entfernt. Der Flächenadapter ergänzt räumlich begrenzte Delaunay-Stützung ohne Lückenfüllung, physische Strahl-/Ebenenverdeckung und gemeinsame XY-Rückprojektion. Optionale Mindestneigung 0° lässt auch flache Stützflächen zu; der vorhandene Fitter-Standard 5° bleibt erhalten. Die aktuelle 295er-Automatik bekommt keine historische starre 160-/6-Archiv-/Externe-Kalibrierungspflicht.

Alle drei PRs waren bei letzter Prüfung Draft/offen/unmerged. Die alte Referenz-Windows-Dateiliste enthielt die neuen SfM-/Graph-/RTK-BA-Tests nicht. `DEV-CI-RGB-WIN-001` ergänzte die sieben vorhandenen Dateien als eigenen Windows-Schritt und den Zielbranch-Pushtrigger; kein nativer Nachweis aus der älteren Referenz-CI ableiten.

**Nachgewiesene Ziel-CI auf `1edde3a21a95ca560d336f2364921de71e38f0ef`:** [Python](https://github.com/multicoptermedia-ship-it/mcm-solarcheck2/actions/runs/40881097022) und [Windows](https://github.com/multicoptermedia-ship-it/mcm-solarcheck2/actions/runs/40881097013) unter 3.11/3.12 erfolgreich, einschließlich des RGB-Schritts. Ein vorheriger Dokumentationslauf scheiterte ausschließlich am überlangen Checkpoint; Kürzung statt Änderung der 8.500-Zeichen-Assertion behob das. Die zwischenzeitlichen Null-Run-/API-Filter-Beobachtungen sind historische Schritte: automatische Workflows und API-Zugang funktionieren inzwischen. Beide Workflows unterstützen zusätzlich `workflow_dispatch`.

Der spätere reine Vorgänger-Dokumentationsstand `ee96a751b223f29c6ddc143f8bb8982aa33e1492` hat ebenfalls erfolgreiche [Python](https://github.com/multicoptermedia-ship-it/mcm-solarcheck2/actions/runs/41055401630)- und [Windows](https://github.com/multicoptermedia-ship-it/mcm-solarcheck2/actions/runs/41055401679)-CI.

Die neue automatische Kamera-/Brennweiten-/Oberflächenentwicklung benötigt eigene Regressionen und CI am endgültigen HEAD; frühere grüne Läufe belegen sie nicht. Lokale Gesamtsuite vor der SIFT-Änderung: 4.056 bestanden, 15 erwartete Skips, 14 Warnungen in 160,20 s; `pip check` und Offline-Qt-Smoke erfolgreich. Der vorhandene positive ZIP-Integrationstest verwendet synthetische JPEGs und Test-Doubles für ORB/SfM/BA; er ist kein echter Flugnachweis. Windows-CI ist keine Windows-10-Hardwareabnahme und keine Orthofoto-Fachfreigabe.

## Cloud-Entwicklung starten

Das getestete Installationsskript und die Startanweisungen sind in der Cloud-Konfiguration gespeichert. In der eingerichteten Maschine:

```bash
cd /workspace/mcm-solarcheck2
. /workspace/.solarcheck-setup/activate.sh
python -m pip check
python -m mcm_solarcheck.offline.windows_smoke --headless
```

Die Venv liegt außerhalb der Repositories unter `/workspace/.venvs/solarcheck-py312`. Caches und Testausgaben ebenfalls außerhalb des Quellbaums halten. Der lokale Offline-Desktop-Smoke erstellt ein temporäres Projekt/SQLite und sperrt Python-Netzwerkverbindungen; er und `pip check` waren erfolgreich. Für native Windows-Starts die vorhandene `INSTALLATION.md` verwenden.

Die Cloud-UI meldete gespeicherte Einstellungen, aber fehlgeschlagene Snapshot-Veröffentlichung. Die aktuelle Netzwerkeinstellung wirkt bereits; Drive ist erreichbar. Die genaue Veröffentlichungsursache ist ohne verfügbare Plattformlogs ungeklärt. Nicht Linux oder die Ordnerfreigabe als Ursache behaupten. Erfolgreicher Betrieb dieser Maschine bestätigt keine Wiederherstellung in einer neuen Aufgabe.

## Vorhandenen Gesamtflugweg fortsetzen

Aufrufhilfe: `python -m mcm_solarcheck.services.rgb_zip_whole_flight_cli --help`. Für jedes vollständige RGB-ZIP/ZIP-kompatible `.tip` ein `--rgb-archive`; für `--report` einen neuen absoluten Pfad außerhalb aller Originalarchivordner wählen. **`--calibration-json` weglassen, um die automatische ungeprüfte Kamerainitialisierung zu verwenden.** Die Originale werden aus ZIP-Streams gelesen, nicht extrahiert oder überschrieben.

```bash
python -m mcm_solarcheck.services.rgb_zip_whole_flight_cli \
  --rgb-archive /absolute/private-inputs/complete-flight.zip \
  --report /absolute/private-output/new-flight-report.json
```

`--candidate-output-dir` ergänzt optional ein **neues absolutes privates Ausgabeverzeichnis außerhalb von Git und Eingabeordnern**; PNG-/NoData-Maskenkandidaten werden erst nach bestandenen vollständigen Geometriegates erzeugt. `--candidate-max-side` und `--candidate-metres-per-pixel` steuern die Auflösung. Die Ausgabe bleibt ein ungeprüfter rekonstruierter Flächenkandidat, keine Qualitätsfreigabe.

Die Beispielpfade stehen für vorhandene lokale Eingaben und ein neues Ausgabeziel. Tatsächliche private Pfade, Manifeste und Zwischenstände werden nur in privaten Cloud-Startanweisungen geführt.

Der erste reale Lauf ohne externe Datei hat alle 295 Merkmalsbilder und 1.958 Paar-Kandidaten verarbeitet und scheiterte im globalen SfM an unzureichender PnP-3D-Stützung. Der frühe Abbruch beim ersten PnP-Fehlschlag wurde korrigiert: weitere gestützte Kameras versuchen, mit unveränderten Positivtiefen-/3-Pixel-Gates und Pflicht aller Bilder. Acht Kameranetztests bestanden. Auch der ORB-Verhältnisfilter-Lauf prüfte 1.921 Geometriepaare in 413,7 s und scheiterte bei 2/295 Kameraposen, 32 Landmarken aus 23.344 Tracks; BA nicht erreicht. Ungefilterter Altbestand scheitert am übernommenen Trackkonfliktgate. Schwellen und Pflicht aller Bilder bleiben unverändert. SIFT ist nun Standard im selben Dienst/CLI (`--feature-method sift`), ORB bleibt explizit über `orb` verfügbar. SIFT nutzt vorhandenes OpenCV, keine neue Abhängigkeit. Ein analytisch gedrehter/beschnittener Texturbild-ZIP-Test zeigt mindestens 45 Matches mit Medianfehler unter 0,5 Pixel; 19 Pipeline-Tests bestanden. Echter SIFT-Lauf aller 295 Bilder mit 2.400 Pixeln/2.400 Merkmalen gestartet, Endergebnis und neue native HEAD-CI noch offen. Kein identischer erneuter Merkmalslauf. Noch keine erfolgreiche vollständige SfM-/BA-/Dachflächen-/Orthofotoausgabe nachgewiesen.

Nächster konkreter Schritt: laufenden vollständigen SIFT-Lauf und neue HEAD-CI auswerten, vollständigen geometrischen Kameragraphen und Reprojektions-/RTK-Hold-out-Residuen auswerten, daraus räumlich gestützte Dachflächen und gemeinsame RGB-Orthorektifizierung erzeugen. Fehlende Flächen bleiben NoData, nicht mit 2D-Stitching füllen. Fenster-BA ist keine simultane globale BA; RTK-Hold-outs sind keine unabhängigen GCPs. Keine ODM-/WebODM-/ODX-Nutzung, Surveyformular- oder KI-Nebenarbeit. Originale/produktive SQLite bleiben unverändert, alle fachlichen Freigaben getrennt.
