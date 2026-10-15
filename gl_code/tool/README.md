# UPS GL Code Filler (for people who don't use Python)

**`GL_Code_Tool.html`** is a single web page. Double-click it to open it in Chrome, Edge or Safari. It works offline and the UPS file never leaves the computer.

## How to use it

1. Open `GL_Code_Tool.html`.
2. Drag the UPS `.csv` file onto the page (or click to choose it). Use the file exactly as exported from UPS: no header row, column M can be empty.
3. Check the **Company** box. It is filled in automatically when the file name contains `CSI`, `Cooper` or `Belimo`; otherwise choose it.
4. Click **Fill GL codes**.
5. Click **Download filled file**. This is your file with column M filled in. Rows the rules are not sure about say `review`.
6. Click **Download review sheet** to get only the `review` rows, with a suggested code and the reason. Open it in Excel, type the right code in the **Final_GL** column, and save it as `.csv`.
7. (Optional) In step 3 of the page, add the filled file and your saved review sheet, and click **Apply my answers**. You get a final file with your codes in column M.

## What the numbers mean

* **filled automatically**: the rules are confident.
* **need a person to check**: written as `review` in column M. A person should decide these.
* **already had a code**: rows that already had a value in column M are never changed.

## For the person who maintains the rules

The rules live in `../GL_Code_Apply.ipynb` (the `RULES_V2` cell). After changing them, rebuild the page:

```
python gl_code/tool/build_tool.py
```

and send the client the new `GL_Code_Tool.html`. The page gives the same answers as the notebook: it was checked row by row against the notebook on the 10/02/26 CSI and Belimo files, and the output csv was byte-identical.

Not included (use the notebook for these): the tracking history ledger across billing periods, and the accuracy check against a completed file.
