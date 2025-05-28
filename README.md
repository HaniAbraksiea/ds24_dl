# Chattbot med RAG

Detta projekt är gjort för kunskapskontrollen i kursen *Deepleraning - Data Scientist EC-utbildning*.  
Jag har skapat en chattbot som använder RAG-teknik för att svara på frågor utifrån innehållet i en magisteruppsats.

---

## 📦 Projektinnehåll

- `Chattbot_med_RAG.ipynb` – huvudnotebooken med kod och evaluering
- `main.py` – terminalversion av chattbotten
- `requirements.txt` – lista över Python-paket som krävs
- `.env` – mall för miljöfil (lägg till din egen API-nyckel)
- `Magisteruppsats.pdf` – dokument som chattbotten använder som källa
- `teoretiska_frågor.txt` – frågor och svar till teoridelen samt självutvärdering

---

## 🔧 Hur man kör

1. Installera paket:
pip install -r requirements.txt

2. Lägg API-nyckeln i `.env`:
API_KEY=ditt_api_key

3. Kör:
- Notebook: öppna `Chattbot_med_RAG.ipynb` i Jupyter
- Terminal: kör `python main.py`

---

## 📈 Evaluering

Boten testas med frågor som:
- ilken typ av intervjuer användes i studien?
- Vilken kommun är studien baserad på?
- Kontrollfrågor där svaret inte finns.

Resultaten visas i notebook och terminal.

---

## 🌍 Reflektion

Jag diskuterar användning, utmaningar och möjligheter i slutet av notebooken.

---

## 👤 Kontakt

Hani Abraksiea
