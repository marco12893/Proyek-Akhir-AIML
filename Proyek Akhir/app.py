from flask import Flask, render_template, request
import pandas as pd
#Import method ataupun variable dari file yg diperlukan
from XGBoost_FA_Cup_3 import accuracyFormatted as accuracyXG_FA, df

from Random_Forest_FA_Cup_3 import accuracyFormatted as accuracyRF_FA
from Logistic_Regression_FA_Cup_3 import accuracyFormatted as accuracyLR_FA
from XGBoost_Premier_League import accuracyFormatted as accuracyXG_PL
from Random_Forest_Premier_League import accuracyFormatted as accuracyRF_PL
from Logistic_Regression_Premier_League import accuracyFormatted as accuracyLR_PL
'''
//CATATAN PERSONAL//
- 0 itu kalah, 1 itu draw, 2 itu menang
- Year yg dipake buat training (PREDIKSI !<= DATA ITU) = 2018 s.d. 2023 
-
- 


Yg harus dikerjakan (Rayner):
- tambahin date dan neutral venue untuk selection & form mengambil dari yang paling recent (date and time, main view)
- tombol stats for nerds di masukin di navbar header
- grafik yg nunjukin progress kita (3 algoritma dari 3 file) (route yg beda)
- classification report (route yg beda juga), boleh tunjukin performance by division gap
- PLOT seberapa penting attribute (route yg beda juga)
- confusion matrix (route yg beda)

- 


(marco):
- modif backend untuk prediksi winner & prediksi goal 
- untuk training data, stats untuk home dan away (berapa win, draw, loss)
- Learning curve (seberapa akurat seiring kita nambahin training data) (route yg beda juga)
- 
-
'''

# XGBoost variables (FA)
formattedXG_FA = "{:.2f}".format(accuracyXG_FA)
# RandomForest variables (FA)
formattedRF_FA = "{:.2f}".format(accuracyRF_FA)
# Logistic_regression variables (FA)
formattedLR_FA = "{:.2f}".format(accuracyLR_FA)
# XGBoost variables (Premier League)
formattedXG_PL = "{:.2f}".format(accuracyXG_PL)
# XGBoost variables (Premier League)
formattedRF_PL = "{:.2f}".format(accuracyRF_PL)
# XGBoost variables (Premier League)
formattedLR_PL = "{:.2f}".format(accuracyLR_PL)

# Ambil team name untuk option bar (masih salah mungkin)
FA_df = df[(df['Type'] == 'FA Cup')]
unique_values = FA_df['Home'].unique()
team_list = unique_values.tolist()



app = Flask(__name__)
# MAIN ROUTE VIEW
@app.route('/')
def home():
    return render_template('starting_page.html',
                           accuracyXG_FAcup=formattedXG_FA,
                           accuracyRF_FAcup=formattedRF_FA,
                           accuracyLR_FAcup=formattedLR_FA,
                           accuracyXG_PL=formattedXG_PL,
                           accuracyRF_PL=formattedRF_PL,
                           accuracyLR_PL=formattedLR_PL,
                           team_list=team_list)

# ignore for now
@app.route('/karema')
def karema():
    pass

if __name__ == '__main__':
    app.run(debug=True)
