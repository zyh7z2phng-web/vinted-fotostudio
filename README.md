# Vinted Fotostudio

Mobile Web-App (PWA): Kleidungsfotos automatisch freistellen, auf reinweißen
Hintergrund (#FFFFFF) setzen, mit weichem zweistufigem Schatten. Alles läuft
im Browser. Die Fotos verlassen das Gerät nicht.

## Projektstruktur

```
index.html              Oberfläche
css/app.css             Design (mobil zuerst, Hell/Dunkel)
js/app.js               Ablauf, Karten, Einstellungen, Speichern/Teilen/ZIP
js/editor.js            Nachbessern: Pinsel, Rückgängig, Pinch-Zoom, Verschieben
js/infer.js             Web Worker: KI-Modell laden (WebGPU/WASM) und freistellen
js/worker.js            Web Worker: Kanten, Farben, Komposition, Masken-Ablage
js/pipeline.js          Bildverarbeitung (reines JS): Maske, Kanten, Farben, Schatten
sw.js                   Service Worker: Offline + Cross-Origin-Isolation
manifest.webmanifest    PWA-Manifest, icons/ = App-Icons
models/                 ISNet (fp16 für WebGPU, int8 für WASM), lokal, kein CDN
vendor/                 onnxruntime-web 1.31.1, JSZip 3.10.2, heic2any 0.0.4
tools/serve.mjs         kleiner lokaler Server
netlify.toml            Netlify-Konfiguration (Header für Multithreading)
```

## Lokal starten

Ein Webserver ist nötig (Service Worker und Module funktionieren nicht über `file://`):

```bash
node tools/serve.mjs 8080
```

Danach http://localhost:8080 öffnen. Alternativ geht `python3 -m http.server 8080`.

Testschalter: `?backend=wasm` erzwingt den WASM-Pfad (so läuft es auf Geräten ohne WebGPU).
`?lowmem=1` bzw. `?lowmem=0` schaltet den Speicher-Sparmodus ein oder aus (auf Handys
standardmäßig an: KI-Worker wird nach jedem Foto neu gestartet, ein Thread).

## Auf dem Handy nutzen (Deployment)

Service Worker, Kamera-Zugriff und „Zum Home-Bildschirm“ brauchen **HTTPS**.
Ein Aufruf über die LAN-IP (http://192.168…) reicht dafür nicht.

**GitHub Pages**
1. Neues Repository anlegen und den gesamten Ordner hochladen (auch `models/`).
   Kein Git LFS verwenden, denn GitHub Pages liefert LFS-Dateien nicht aus.
   Die größte Datei ist 88 MB und liegt damit unter dem 100-MB-Limit von GitHub
   (GitHub warnt ab 50 MB, das ist unkritisch).
2. Settings → Pages → „Deploy from a branch“ → `main` / `root`.
3. Nach ein bis zwei Minuten unter `https://<name>.github.io/<repo>/` öffnen.

**Netlify**
- Ordner auf https://app.netlify.com/drop ziehen oder das Repository verbinden
  (kein Build-Befehl, Publish-Verzeichnis `.`). `netlify.toml` setzt die
  COOP/COEP-Header, dann laufen die WASM-Threads schon beim ersten Laden.

**Auf dem iPhone installieren:** Seite in Safari öffnen → Teilen → „Zum
Home-Bildschirm“. Beim ersten Start wird das KI-Modell einmalig geladen
(44 MB, bei WebGPU 88 MB). Danach funktioniert die App offline. Als
Home-Bildschirm-App ist der Speicher vor Safaris automatischer Bereinigung geschützt.

## Bedienung

- **Fotos hinzufügen**: mehrere auf einmal, aus Galerie oder Kamera. HEIC wird
  unterstützt (Safari nativ, sonst über heic2any).
- **Einstellungen**: Format 3:4 (1200×1600) oder Quadrat (1200×1200), Rand
  (Standard 8 %), Schatten 0-100 % (Standard 35 % = dezent), automatische
  Helligkeit/Weißabgleich (standardmäßig aus). Änderungen wirken sofort auf alle Fotos.
- Pro Foto: **Original/Ergebnis**, **Nachbessern** (Vollbild: Radieren/Zurückholen,
  Pinselgröße, Rückgängig, Pinch-Zoom und Zwei-Finger-Verschieben; am Rechner
  Mausrad, Alt/Shift-Ziehen), **Speichern**, **Löschen**.
- **Speichern** auf dem iPhone öffnet das Teilen-Menü → „Bild sichern“ legt das
  Foto in der Fotos-App ab. **Alle sichern** teilt alle Bilder auf einmal,
  **Alle als ZIP** erzeugt ein ZIP („In Dateien sichern“). Am Rechner werden die
  Dateien heruntergeladen. JPEG-Qualität 0,92.

## Wie die Freistellung funktioniert

1. Foto wird mit korrekter EXIF-Drehung dekodiert und intern auf max. 1500 px
   verkleinert.
2. **ISNet** (Dichotomous Image Segmentation) berechnet bei 1024×1024 eine
   Alpha-Maske, mit WebGPU (fp16) oder WASM mit SIMD und Threads (int8).
3. Aufräumen: kleine, abgetrennte Flecken und Restschleier werden entfernt;
   mehrere große Teile (z. B. zwei Schuhe) bleiben erhalten.
4. Für den Export wird der Produktausschnitt **aus dem Originalfoto** in voller
   benötigter Auflösung geholt (nicht aus dem 1500-px-Arbeitsbild) → scharfes Ergebnis.
5. Kanten: begrenzter Farb-Guided-Filter (legt die Kante an echte Bildkanten,
   darf aber kaum Fläche hinzufügen), ca. 1 px Kantenzug nach innen, dann
   **Farbdekontamination** (Blur-Fusion-Vordergrundschätzung nach Forte &
   Pitié 2021). Sie entfernt die alte Hintergrundfarbe aus halbtransparenten
   Randpixeln, sodass kein heller oder dunkler Saum entsteht.
6. Komposition: Zuschnitt auf die Bounding Box, zentriert mit Rand; Schatten aus
   der Silhouette: enger Kontaktschatten plus großer weicher Schatten, beide
   leicht nach unten versetzt.

## Bekannte Grenzen

- **Weiß auf Weiß** (weißes Teil auf weißem Laken): Kanten sind meist gut,
  aber sehr helle Teile (z. B. weiße Socken, weiße Innensohle) gehen teils
  verloren → mit „Zurückholen“ nachmalen. Dunkle oder strukturierte Unterlage
  hilft am meisten.
- **Ein Teil pro Foto**: Liegen mehrere Teile im Bild (Outfit-Flatlay), wählt
  das Modell das auffälligste aus. Den Rest per Pinsel ergänzen oder entfernen.
- **Kleiderbügel** werden als Teil des Produkts erkannt → bei Bedarf wegradieren.
- **Körniger Untergrund** (Kies, Teppich): an flauschigen Rändern können
  einzelne Körner hängen bleiben.
- Feine Haare/Fransen werden weich, aber nicht haargenau freigestellt (die
  Modellauflösung beträgt 1024 px).
- Erster Start: 44-88 MB Download. Auf älteren iPhones ohne WebGPU dauert ein
  Foto mehrere Sekunden (WASM).
- Nur ein Bearbeitungsstand im Speicher: Beim Neuladen der Seite sind die Fotos
  weg (bewusst, nichts wird gespeichert).

## Lizenzen

- ISNet-Gewichte (onnx-community/ISNet-ONNX, Basis IS-Net/DIS, Konvertierung
  von IMG.LY): **AGPL-3.0**. Für die eigene Nutzung unproblematisch. Wer die App
  öffentlich anbietet, muss den Quellcode unter AGPL zugänglich machen (z. B.
  öffentliches GitHub-Repo).
- onnxruntime-web: MIT · JSZip: MIT/GPLv3 · heic2any: MIT.
