# Externe RGB-Kamerakalibrierung – Anforderungen

Stand: 30.10.2026. Nutzerentscheidung: **zuerst externe Kalibrierung beschaffen**, keinen neuen Selbstkalibrierungsmodus entwickeln. Maßgeblich bleibt [AKTUELLER_ENTWICKLUNGSSTAND.md](AKTUELLER_ENTWICKLUNGSSTAND.md).

## Welche Kamera und Bilder?

Eine repräsentative, tatsächlich heruntergeladene Originaldatei nennt `Make=DJI`, `Model=M3T` und enthält das RGB-Hauptbild mit **4000×3000 Pixeln** im JPEG-/MPO-Container. Gesucht ist die Kalibrierung der **RGB-Kamera für diese Originalausgabe**, keine Thermal-/Temperaturkalibrierung.

Die Kalibrierung muss zum konkreten Kameragerät, Objektiv und verwendeten Aufnahme-/Ausgabemodus passen. Auflösung, Seitenverhältnis, Ausschnitt/Binning, Zoom-/Fokuseinstellung und bereits angewandte kamerainterne Entzerrung müssen geklärt sein. Gleiche Pixelabmessungen allein beweisen diese Übereinstimmung nicht. Seriennummern und gerätebezogene Unterlagen nur privat austauschen, nicht in GitHub ablegen.

## Benötigte Werte und Nachweise

| Bestandteil | Benötigte Angaben |
| --- | --- |
| Intrinsics | `fx`, `fy`, `cx`, `cy` in **Pixeln des 4000×3000-Originalrasters**, samt Koordinatenkonvention. Das bestehende Modell verwendet eine Kameramatrix ohne Scherung. |
| Verzeichnung | Koeffizienten mit eindeutigem Modell, Reihenfolge und Konvention. Der bestehende OpenCV-Pfad unterstützt 4/5/8/12/14 Werte. |
| Bildverarbeitung | Angabe, ob die ausgegebenen JPEGs bereits entzerrt sind und für welchen konkreten Bildzustand die Parameter gelten. Nullkoeffizienten nicht ohne Nachweis einsetzen. |
| Zuordnung | Welches Gerät/Objektiv und welcher Aufnahme-/Ausgabemodus wurden kalibriert? Für welche Auflösung gelten die Werte? |
| Provenienz | Aussteller, Datum, Methode und nachvollziehbare Kalibrierungsunterlagen. |
| Validierung | Dokumentierte Fehler-/Genauigkeitsbewertung und möglichst separate Prüfbilder oder ein unabhängiger Prüfbericht. Ein kleines Anpassungsresiduum der verwendeten Kalibrierbilder allein ist kein unabhängiger Nachweis. |

Allgemeine Datenblatt-Brennweite, Kleinbildäquivalent, GNSS-/RTK-Daten, Gimbalwinkel oder die bloße Übereinstimmung der RGB-/Thermal-Bildnummern ersetzen diesen Nachweis nicht. Der aktuelle Bestand ist noch nicht photogrammetrisch rekonstruiert.

## Praktischer Beschaffungsweg

1. Beim Lieferanten beziehungsweise [DJI Enterprise Support](https://www.dji.com/support/product/mavic-3-enterprise) gezielt nach einer zu diesem Gerät und JPEG-Ausgabemodus passenden RGB-Kalibrierung fragen. Ein weiterer offizieller Einstieg ist [DJI Service](https://www.dji.com/support/repair).
2. Es ist bisher nicht belegt, dass DJI solche gerätespezifischen Dateien öffentlich oder auf Anfrage bereitstellt. Die [offizielle Produkt-Downloadseite](https://enterprise.dji.com/mavic-3-enterprise/downloads) enthält keine hier nachgewiesene passende RGB-Intrinsics-Datei. ThermalSDK ist kein Ersatz für RGB-Kalibrierung.
3. Falls kein passender Hersteller-/Lieferantennachweis verfügbar ist: ein fachkundiges Kamerakalibrierungs-/Photogrammetrielabor mit genau diesem Gerät und passendem Bildmodus beauftragen. Vorher klären, ob dessen Kalibrierung auf die bereits vorhandenen Aufnahmen anwendbar ist. Kalibrierbilder dürfen getrennt aufgenommen werden; Originalflugaufnahmen bleiben unverändert.

Es wurden keine Anbieter kontaktiert, Aufträge erteilt oder Kosten ausgelöst.

## Kurzer Anfragetext

> Für DJI M3T-RGB-Originalaufnahmen mit 4000×3000 Pixeln benötigen wir eine gerätebezogene Kamerakalibrierung. Können Sie passende Intrinsics (`fx`, `fy`, `cx`, `cy` in Originalpixeln), Verzeichnungsparameter mit Modell/Konvention sowie einen Kalibrierungs- und Validierungsnachweis bereitstellen? Bitte bestätigen Sie den unterstützten Aufnahme-/Zoom-/Fokusmodus und ob die Parameter für die bereits ausgegebenen, gegebenenfalls kameraintern entzerrten JPEG-/MPO-Bilder gelten. Wenn keine solche Datei verfügbar ist: Welche Kalibrierungsleistung und Unterlagen können Sie für dieses konkrete Gerät anbieten?

## Übernahme in SolarCheck 2

Herstellerdatei, OpenCV-YAML/XML oder Bericht zunächst privat zur Prüfung bereitstellen. Niemand muss ungeprüft ein SolarCheck-JSON ausfüllen. Passt das Modell, werden die echten Werte nachvollziehbar in den vorhandenen Vertrag `mcm.rgb_camera_intrinsics_external.v1` übertragen. Der CLI liest derzeit ausschließlich dieses JSON; ein allgemeiner Hersteller-Konverter ist noch nicht vorhanden.

`lens_calibration_verified` darf erst bei tatsächlich geprüftem passenden Nachweis gesetzt werden. Keine Zahlen aus Beispieltests oder nominalen Millimeter-Brennweiten übernehmen und kein Prüfmerkmal zum Umgehen des Eingangsschutzes setzen. Erst danach die lokal vorbereitete vollständige RGB-Serie im bestehenden SfM-/RTK-BA-Pfad verarbeiten. Vermessungs-, Dach-/Orthofoto-, PV- und Thermal-Freigaben bleiben separate Gates.
