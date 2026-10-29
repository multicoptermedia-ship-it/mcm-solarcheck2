# SolarCheck 2 – Entwicklung im eigenen Zielrepository

Stand: 29.10.2026. Zuerst `AKTUELLER_ENTWICKLUNGSSTAND.md` lesen. Diese Datei ergänzt den übernommenen Checkpoint um die ausdrückliche Repository-Entscheidung des Nutzers und die Cloud-Einrichtung.

## Repository und übernommener Stand

- Ausschließliches Entwicklungsziel: `multicoptermedia-ship-it/mcm-solarcheck2`.
- Entwicklungsbranch: `development/rgb-photogrammetry`; keine Schreibfreigabe für `main`.
- Übernommene Codebasis: `6b72e8f373c3eb8f61615916705699b048616531` aus dem Referenzbranch `feature/orthofoto-license-neutral-rgb-first-20261028`.
- `multicoptermedia-ship-it/MCM-SolarCheck` einschließlich seines lokalen Checkouts bleibt ausschließlich lesend. Dort keine Änderungen, Commits, Pushes, PR-Änderungen oder Merges.
- Die Übernahme erhält die vorhandene Historie und implementiert keine bereits fertigen Komponenten erneut. Dieser Übernahmeschritt ändert keine Anwendungslogik, Tests oder Abhängigkeiten und gibt kein Orthofoto frei.

## Bereits verifizierte Ergebnisse

Die bestehenden GitHub-Checks wurden am 29.10.2026 über die Actions- und PR-Checks-Seiten gelesen, nicht neu im Referenzrepository gestartet:

| Referenz-PR | Vollständiger HEAD | Python-CI | Windows-CI |
| --- | --- | --- | --- |
| #7 | `14842d98ed00c4c6f9416dc3157149ce4d0ac721` | [erfolgreich](https://github.com/multicoptermedia-ship-it/MCM-SolarCheck/actions/runs/40561014910) | [erfolgreich](https://github.com/multicoptermedia-ship-it/MCM-SolarCheck/actions/runs/40561014896) |
| #8 | `2f1ac0e967415308b7ab97e78e83d5b32d99f7c6` | [erfolgreich](https://github.com/multicoptermedia-ship-it/MCM-SolarCheck/actions/runs/40562888287) | [erfolgreich](https://github.com/multicoptermedia-ship-it/MCM-SolarCheck/actions/runs/40562889517) |
| #9 | `6b72e8f373c3eb8f61615916705699b048616531` | [erfolgreich](https://github.com/multicoptermedia-ship-it/MCM-SolarCheck/actions/runs/40765128985) | [erfolgreich](https://github.com/multicoptermedia-ship-it/MCM-SolarCheck/actions/runs/40765129095) |

Alle drei PRs waren weiterhin Draft/offen. Python- und Windows-Matrizen verwenden 3.11/3.12; Windows-CI ist keine Windows-10-Hardwareabnahme. Damit ist die im übernommenen Checkpoint noch offene CI-Prüfung für PR #9 am genannten HEAD geklärt. Neue Commits benötigen ihre eigenen CI-Belege.

Lokal im Zielcheckout wurden die bestehenden RGB-/SfM-/Graph-/BA-/RTK- und Offline-Starttests unter Python 3.12 ausgeführt: **37 bestanden, 2 native Windows-Tests unter Linux übersprungen**. `pip check` und der Offline-Desktop-Smoke mit temporärer SQLite und gesperrten Python-Netzwerkverbindungen waren erfolgreich. Der positive Ganzflug-Integrationstest ersetzt ORB/SfM/BA durch Test-Doubles; daraus folgt keine echte 160-Bild-Rekonstruktion.

## Cloud-Entwicklung starten

Die Cloud-Umgebung speichert das vollständige getestete Installationsskript und die Startanweisungen in ihrer Konfiguration. In der eingerichteten Maschine:

```bash
cd /workspace/mcm-solarcheck2
. /workspace/.solarcheck-setup/activate.sh
python -m pip check
python -m mcm_solarcheck.offline.windows_smoke --headless
```

Die Venv liegt außerhalb der Repositories unter `/workspace/.venvs/solarcheck-py312`. Testausgaben und Caches gehören ebenfalls außerhalb des Quellbaums. Für native Windows-Starts die bereits vorhandenen Anweisungen in `INSTALLATION.md` verwenden. Keine zusätzlichen Worktrees anlegen; jede Cloud-Aufgabe ist bereits isoliert.

## Unveränderter nächster fachlicher Schritt

Die vorhandene `mcm_solarcheck.services.rgb_zip_whole_flight_cli` mit der vollständigen privaten Flugserie read-only ausführen. Benötigt werden alle sechs RGB-ZIP-Archive einschließlich der gültigen ZIP-Datei `RGB2.tip` mit insgesamt 160 Hauptaufnahmen sowie eine unabhängig verifizierte Kamerakalibrierung. In dieser Cloud-Maschine wurden diese privaten Eingaben noch nicht bereitgestellt. Keine erneute ZIP-/EXIF-/MPO-/Bildpaar-Inventur anfordern oder entwickeln.

Der vorhandene Kalibrierungsvertrag ist `mcm.rgb_camera_intrinsics_external.v1`: echte geprüfte Intrinsics `fx/fy/cx/cy` in Pixeln der Originalbilder, dazu passende OpenCV-Verzeichnungskoeffizienten und nachvollziehbare Kalibrierungsprovenienz. Das Prüfmerkmal `lens_calibration_verified` darf ausschließlich bei tatsächlich unabhängig geprüfter Kalibrierung gesetzt werden. Keine Werte aus der 4,4-mm-EXIF-Brennweite erfinden; synthetische Testwerte sind keine DJI-Kalibrierung.

Aufrufhilfe: `python -m mcm_solarcheck.services.rgb_zip_whole_flight_cli --help`. Für jedes Archiv `--rgb-archive` angeben, die verifizierte Datei mit `--calibration-json` übergeben und für `--report` einen neuen absoluten Pfad außerhalb aller Originalarchivordner wählen. Originale und produktive SQLite bleiben unverändert; private Bilder, Koordinaten, Kalibrierungsunterlagen und Kundendaten gehören nicht in GitHub.

Anschließend den vollständigen geometrischen Kameragraphen, Reprojektionsfehler und interne RTK-Hold-out-Residuen auswerten. Erst mit diesem Beleg folgen rekonstruierte Dachflächen und RGB-Orthorektifizierung, daraus PV-Modulprüfung, anschließend Thermal. Der bestehende Lauf liefert zunächst einen unbestätigten RTK-/SfM-Kandidaten; keine vollständige globale BA, kein geprüftes Dachmodell und kein verifiziertes Orthofoto sind damit behauptet. RTK-Hold-outs sind keine unabhängigen GCPs. ODM/WebODM/ODX bleiben ausgeschlossen.
