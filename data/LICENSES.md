# רישיונות המקורות

נכתב אוטומטית על ידי `src/fetch_sefaria.py` בזמן המשיכה.

| מקור | מהדורה | רישיון | מסחרי? | מקטעים | דוגמה |
|---|---|---|---|---|---|
| Mishnah | Torat Emet 357 | Public Domain | כן | 524 | `Mishnah Berakhot 1` |
| Mishnah | Version of the Gemara - vocalized | Public Domain | כן | 1 | `Mishnah Bikkurim 4` |
| Halakhah/Mishneh Torah | Friedberg Edition | Public Domain | כן | 39 | `Kuntres Zikah 1` |
| Halakhah/Mishneh Torah | Mishneh Torah with Commentary by Rabbi Adin Even-Israel Steinsaltz, Koren Publishers | CC-BY-NC | **לא** | 84 | `Steinsaltz Introductions to Mishneh Torah, Foundations of the Torah` |
| Halakhah/Mishneh Torah | Torat Emet 363 | Public Domain | כן | 954 | `Mishneh Torah, Foundations of the Torah 1` |
| Halakhah/Mishneh Torah | Torat Emet 370 | Public Domain | כן | 51 | `Mishneh Torah, Reading the Shema 1` |
| Halakhah/Mishneh Torah | Wikisource Mishneh Torah | CC-BY-SA | כן | 18 | `Mishneh Torah, Transmission of the Oral Law` |
| Siddur Ashkenaz | Daat Siddur Ashkenaz | Public Domain | כן | 218 | `Siddur Ashkenaz, Shabbat, Shacharit, Preparatory Prayers, Modeh Ani` |
| Siddur Ashkenaz | Siddur Veshinantam  | unknown | **לא ידוע** | 1 | `Siddur Ashkenaz, Weekday, Maariv, Additions for Motza'ei Shabbat, Veyiten Lekha` |
| Siddur Ashkenaz | Tanach with Nikkud | unknown | **לא ידוע** | 1 | `Siddur Ashkenaz, Weekday, Shacharit, Post Service, Ten Commandments` |
| Siddur Ashkenaz | The Metsudah siddur, 1981 | CC-BY | כן | 221 | `Siddur Ashkenaz, Weekday, Shacharit, Preparatory Prayers, Modeh Ani` |
| Siddur Ashkenaz | The Metsudah siddur: a new linear siddur with English translation by Avrohom Davis, 1981 | CC-BY | כן | 7 | `Siddur Ashkenaz, Festivals, Rosh Chodesh, Hallel, Psalm 113` |
| Siddur Ashkenaz | Torat Emet 357 | unknown | **לא ידוע** | 1 | `Siddur Ashkenaz, Shabbat, Kabbalat Shabbat, Bameh Madlikin` |
| Bavli | William Davidson Edition - Vocalized Aramaic | CC-BY-NC | **לא** | 5349 | `Berakhot 2a` |
| Tanakh | Miqra according to the Masorah | CC-BY-SA | כן | 929 | `Genesis 1` |

## אזהרה

המהדורות הבאות אינן מתאימות למוצר או שירות מסחרי:

- **Halakhah/Mishneh Torah** (Mishneh Torah with Commentary by Rabbi Adin Even-Israel Steinsaltz, Koren Publishers) — CC-BY-NC
- **Bavli** (William Davidson Edition - Vocalized Aramaic) — CC-BY-NC

למהדורות הבאות ספריא לא החזירה רישיון. הן נחשבות אסורות לשימוש מסחרי עד שיוכח אחרת:

- **Siddur Ashkenaz** (Siddur Veshinantam )
- **Siddur Ashkenaz** (Tanach with Nikkud)
- **Siddur Ashkenaz** (Torat Emet 357)


לבניית מסלול מסחרי: `python src/fetch_sefaria.py --commercial-safe`.
