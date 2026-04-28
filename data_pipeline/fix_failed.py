# data_pipeline/fix_failed.py
import sys
import os

# Tell Python: "project root is one level up from this file"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CORRECTIONS = {
    "INFOSYS"    : "INFY",
    "TATAMOTORS" : "TATAMOTORS",
    "ADANITRANS" : "ADANITRANS",
    "MCDOWELL-N" : "MCDOWELL-N",
    "ZOMATO"     : "ZOMATO",
    "GMRINFRA"   : "GMRAIRPORT",
    "MINDTREE"   : "LTIM",
}

from data_pipeline.crawler import run_crawler

fixed_tickers = list(CORRECTIONS.values())
run_crawler(tickers=fixed_tickers, max_workers=7)