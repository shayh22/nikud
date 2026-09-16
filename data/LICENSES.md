# רישיונות המקורות

נכתב אוטומטית על ידי `src/fetch_sefaria.py` בזמן המשיכה.

| מקור | מהדורה | רישיון | מסחרי? | מקטעים | דוגמה |
|---|---|---|---|---|---|
| Mishnah | Torat Emet 357 | Public Domain | כן | 524 | `Mishnah Berakhot 1` |
| Mishnah | Version of the Gemara - vocalized | Public Domain | כן | 1 | `Mishnah Bikkurim 4` |
| Siddur Ashkenaz | Daat Siddur Ashkenaz | Public Domain | כן | 218 | `Siddur Ashkenaz, Shabbat, Shacharit, Preparatory Prayers, Modeh Ani` |
| Siddur Ashkenaz | Siddur Veshinantam  | unknown | כן | 1 | `Siddur Ashkenaz, Weekday, Maariv, Additions for Motza'ei Shabbat, Veyiten Lekha` |
| Siddur Ashkenaz | Tanach with Nikkud | unknown | כן | 1 | `Siddur Ashkenaz, Weekday, Shacharit, Post Service, Ten Commandments` |
| Siddur Ashkenaz | The Metsudah siddur, 1981 | CC-BY | כן | 221 | `Siddur Ashkenaz, Weekday, Shacharit, Preparatory Prayers, Modeh Ani` |
| Siddur Ashkenaz | The Metsudah siddur: a new linear siddur with English translation by Avrohom Davis, 1981 | CC-BY | כן | 7 | `Siddur Ashkenaz, Festivals, Rosh Chodesh, Hallel, Psalm 113` |
| Siddur Ashkenaz | Torat Emet 357 | unknown | כן | 1 | `Siddur Ashkenaz, Shabbat, Kabbalat Shabbat, Bameh Madlikin` |
| Bavli | William Davidson Edition - Vocalized Aramaic | CC-BY-NC | **לא** | 1652 | `Berakhot 2a` |
| Tanakh | Miqra according to the Masorah | CC-BY-SA | כן | 929 | `Genesis 1` |

## אזהרה

המקורות הבאים אינם מתאימים למוצר או שירות מסחרי:

- **Bavli** (William Davidson Edition - Vocalized Aramaic) — CC-BY-NC

לבניית מסלול מסחרי: `python src/fetch_sefaria.py --commercial-safe`.
