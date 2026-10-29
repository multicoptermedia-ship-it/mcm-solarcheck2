# Verbindliche Repository-Grenze

Nutzeranweisung vom 29.10.2026:

- Alle Entwicklungsänderungen gehören ausschließlich in `multicoptermedia-ship-it/mcm-solarcheck2` (lokal: `/workspace/mcm-solarcheck2`). Das gilt für Quellcode, Tests, Dokumentation, CI-Konfiguration, Commits, Branches und Pull Requests.
- `multicoptermedia-ship-it/MCM-SolarCheck` ist ausschließlich eine lesende Referenz. Dort sind keine Schreibaktionen erlaubt: keine Dateiänderungen, Commits, Pushes, Branch-/Tag-Änderungen, PR-Änderungen oder Merges.
- Auch den lokalen Referenzcheckout `/workspace/MCM-SolarCheck` unverändert lassen. Keine generierten Dateien oder geänderten Git-Metadaten dort erzeugen. Zum Aktualisieren von Referenzdaten bei Bedarf im Zielcheckout vom Remote `reference` lesen/fetchen; keine Änderungen zum Referenzremote pushen.
- Vor jeder Schreibaktion oder Veröffentlichung Repository, Arbeitsverzeichnis und Zielremote prüfen. Historische Arbeitsanweisungen in übernommenen Dokumenten, die Schreiben im alten Repository vorsehen, sind durch diese Nutzeranweisung ersetzt.
- Vorhandene Entwicklungsstände und Testergebnisse wiederverwenden. Fachliche Reihenfolge: eigene Mehrbild-Photogrammetrie für vollständige RGB-Orthofotos, daraus PV-Modulprüfung, anschließend Thermal.
- Nach `docs/AKTUELLER_ENTWICKLUNGSSTAND.md` auch `docs/SOLARCHECK2_START.md` lesen. Dort stehen die Übernahmebasis, die bereits geprüfte Referenz-CI und die offenen Eingaben für den echten Gesamtfluglauf.
- Keine Schreibfreigabe für `main` aus der Wahl des Zielrepositorys ableiten. Bestehende Schutzvorgaben für Originalbilder, produktive SQLite-Dateien und fachliche Freigabegates bleiben bestehen.
