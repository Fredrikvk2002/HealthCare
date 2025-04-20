# make_expansion_status.py
import csv

# 1) List of the 50 states + DC, by FIPS code
statefips = [
    "01","02","04","05","06","08","09","10","11","12","13","15","16","17","18",
    "19","20","21","22","23","24","25","26","27","28","29","30","31","32","33",
    "34","35","36","37","38","39","40","41","42","44","45","46","47","48","49",
    "50","51","53","54","55","56"
]

# 2) ACA Medicaid‐expansion adoption year, from KFF/CMS.
#    States that never expanded get 999 (so exp_flag stays 0)
expansion_year = {
    # 2014 expansions
    "06":"2014","21":"2014","23":"2014","24":"2014","39":"2014","42":"2014",
    "44":"2014","51":"2014","53":"2014","54":"2014",
    # 2015 expansions
    "26":"2015","38":"2015","45":"2015","49":"2015",
    # 2016 expansions
    "28":"2016","37":"2016","55":"2016",
    # 2017 expansions
    "12":"2017","29":"2017","30":"2017","50":"2017",
    # 2018 expansions
    "17":"2018","47":"2018",
    # 2019 expansions
    "41":"2019","46":"2019",
    # 2020 expansions
    "31":"2020","02":"2020","20":"2020",
    # all others never expanded
}

years = list(range(2010, 2024))

with open("expansion_status.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["statefip", "year", "expansion", "post"])
    for s in statefips:
        ay = expansion_year.get(s, 999)
        for y in years:
            exp_flag = 1 if (ay != 999 and y >= int(ay)) else 0
            post_flag = 1 if y >= 2014 else 0
            w.writerow([s, y, exp_flag, post_flag])

print("✔️  expansion_status.csv written (51 × 14 = 714 rows)")
