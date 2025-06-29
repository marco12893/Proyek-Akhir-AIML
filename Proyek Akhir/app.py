from flask import Flask, render_template, request,jsonify,Response
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
import io

# Import method ataupun variable dari file yg diperlukan
from XGBoost_FA_Cup_3 import accuracyFormatted as accuracyXG_FA, df, confidence_df as results_XG_FA, metrics_data, division_gap, plot_data as plot_dataXG_FA, cm as cmXG_FA, confusion_plot as confusion_plotXG_FA, predict_match_score as predict_match_fc_xg
from XGBoost_FA_Cup_2 import accuracyFormatted as accuracyXG_FA_second
from XGBoost_FA_Cup import accuracyFormatted as accuracyXG_FA_first

from Random_Forest_FA_Cup_3 import accuracyFormatted as accuracyRF_FA, confidence_df as results_RF_FA, metrics_data as metrics_data_RF_FA, division_gap as division_gap_RF_FA, plot_data as plot_dataRF_FA, cm as cmRF_FA, confusion_plot as confusion_plotRF_FA, predict_match_score as predict_match_fc_rf
from Random_Forest_FA_Cup_2 import accuracyFormatted as accuracyRF_FA_second
from Random_Forest_FA_Cup import accuracyFormatted as accuracyRF_FA_first

from Logistic_Regression_FA_Cup_3 import accuracyFormatted as accuracyLR_FA, confidence_df as results_LR_FA, metrics_data as metrics_data_LR_FA, division_gap as division_gap_LR_FA, plot_data as plot_dataLR_FA, cm as cmLR_FA, confusion_plot as confusion_plotLR_FA, predict_match_logistic as predict_match_fc_lr, predict_match_score_logistic as predict_match_score_fc_lr
from Logistic_Regression_FA_Cup_2 import accuracyFormatted as accuracyLR_FA_second
from Logistic_Regression_FA_Cup import accuracyFormatted as accuracyLR_FA_first

from XGBoost_Premier_League import accuracyFormatted as accuracyXG_PL, predict_match_premier_league as predict_match_premier_league_xg, predict_match_score_premier_league as predict_match_score_premier_league_xg
from Random_Forest_Premier_League import accuracyFormatted as accuracyRF_PL, predict_match_premier_league_rf, predict_match_score_premier_league_rf
from Logistic_Regression_Premier_League import accuracyFormatted as accuracyLR_PL
'''
//CATATAN PERSONAL//
- 0 itu kalah, 1 itu draw, 2 itu menang
- Year yg dipake buat training (PREDIKSI !<= DATA ITU) = 2018 s.d. 2023 
- Kata kata yang ditaruh di dalam tanda kurung () itu komentar dari Marco
- 


Yg harus dikerjakan (Rayner):
- tambahin date dan neutral venue untuk selection & form mengambil dari yang paling recent (date and time, main view)
- [SUDAH] tombol stats for nerds di masukin di navbar header
- [SUDAH] grafik yg nunjukin progress kita (3 algoritma dari 3 file, totalnya 9 ya) (route yg beda)
- [SUDAH] classification report (route yg beda juga)
- [SUDAH] Tunjukin performance by division gap
- [SUDAH] PLOT seberapa penting attribute (route yg beda juga)
- [SUDAH] confusion matrix (route yg beda)


(marco):
- [SUDAH] modif backend untuk prediksi winner 
- prediksi goal 
- Learning curve (seberapa akurat seiring kita nambahin training data) (route yg beda juga)


(Fabio):
- Betulin prediksi winner untuk FA Cup biar gak ada drawnya 
- (SUDAHH) Betulin feature engineeringnya supaya form yang ditampilin itu udah bener 
  ini berpotensi naikin akurasi prediksi kita dengan buanyak kalau sukses
  
(Chris): 
- (DONE) progress bar & small styling
- Update & Sambungin dashboard dgn backend
- Pindahin tabel ke stats for nerds, tambahin bar win rate and lose rate (winrate lose rate ini buat apa ya? Aku gk paham)
- Tambahain logo untuk semua team (Gak terlalu penting ini, lakuin kalau udah ada waktu aja)
- Tambahin search bar di drop down  
  
  
//** NEW TO-DO LIST **//:

\\NOTE = predict_match_score() itu function buat ambil score prediction\\ 
SUDAH:
- pindahin confidence (5 recent) to statsfornerds
- premier league udh ada prediction team + accuracy
- normalize pada model accuracy di hilangin
- plot confusion matrix (stats for nerds) (aku gk ketemu plot confusion matrixnya marco jadi aku buat heatmap sendiri)
- selection buat prediction yang mana (mau prediction FA CUP atau Premier league)
- Precentage win + team yang menang (win loss bar)
- prediction score buat FA dan Premier league (baru XGBoost_FA_Cup_3)

BELUM:
- Sambungin front end ke back end. Form yang di display harus sesuai dengan tim dan formnya pada tanggal itu. Selain itu confidence barnya, accuracynya harus sesuai dengan apa yang di outputkan oleh prediction function.
- Tambahin statistik gol ke stats for nerds. Masukin seperti RMSE, MSE, direction accuracy, exact, accuracy, within 1 score accuracy.
- Tambahin Display untuk nentuin division berapa timnya itu. Misal divisi 1,2,3,4,5,atau 6.
- warna navbar, body, card bisa dibuat lebih formal (jgn putih polos doang)
'''


# XGBoost variables (FA)
results_XG_FA  = results_XG_FA.to_dict(orient='records')
formattedXG_FA = "{:.2f}".format(accuracyXG_FA)
formattedXG_FA_second = "{:.2f}".format(accuracyXG_FA_second)
formattedXG_FA_first = "{:.2f}".format(accuracyXG_FA_first)

# RandomForest variables (FA)
results_RF_FA  = results_RF_FA.to_dict(orient='records')
formattedRF_FA = "{:.2f}".format(accuracyRF_FA)
formattedRF_FA_second= "{:.2f}".format(accuracyRF_FA_second)
formattedRF_FA_first = "{:.2f}".format(accuracyRF_FA_first)

# Logistic_regression variables (FA)
results_LR_FA = results_LR_FA.to_dict(orient='records')
formattedLR_FA = "{:.2f}".format(accuracyLR_FA)
formattedLR_FA_second = "{:.2f}".format(accuracyLR_FA_second)
formattedLR_FA_first = "{:.2f}".format(accuracyLR_FA_first)

# XGBoost variables (Premier League)
formattedXG_PL = "{:.2f}".format(accuracyXG_PL)

# XGBoost variables (Premier League)
formattedRF_PL = "{:.2f}".format(accuracyRF_PL)

# XGBoost variables (Premier League)
formattedLR_PL = "{:.2f}".format(accuracyLR_PL)

app = Flask(__name__)

def get_team_form(team_name, num_matches=5):
    # Filter matches where the team was home or away
    team_matches = df[(df['Home'] == team_name) | (df['Away'] == team_name)]

    # Sort by date DESCENDING
    team_matches = team_matches.sort_values(by='Date', ascending=False)

    # Get only recent N matches
    recent_matches = team_matches.head(num_matches)

    form = []

    for _, row in recent_matches.iterrows():
        if row['Home'] == team_name:
            if row['Winner'] == 2:  # Home win
                form.append('W')
            elif row['Winner'] == 1:
                form.append('D')
            else:
                form.append('L')
        else:  # team is Away
            if row['Winner'] == 0:  # Away win
                form.append('W')
            elif row['Winner'] == 1:
                form.append('D')
            else:
                form.append('L')

    return form

# MAIN ROUTE VIEW
@app.route('/', methods=['GET', 'POST'])
def home():
    date = []
    team1 = None
    team2 = None
    form_team1 = []
    form_team2 = []
    team_list=[]
    selected_type = request.args.get('type')
    accuracyXG = None
    accuracyRF = None
    accuracyLR = None
    prediction_xg = None
    prediction_rf = None
    prediction_lr = None

    # Get team
    if selected_type == 'fc':
        FA_df = df[(df['Type'] == 'FA Cup')]
        unique_values = FA_df['Home'].unique()
        unique_values.sort()
        team_list = unique_values.tolist()
    elif selected_type == 'pl':
        FA_df = df[(df['Type'] == 'League')]
        unique_values = FA_df['Home'].unique()
        unique_values.sort()
        team_list = unique_values.tolist()

    if request.method == 'GET':
        date = request.args.get('date')
        team1 = request.args.get('team1')
        team2 = request.args.get('team2')

        if team1:
            form_team1 = get_team_form(team1)
        if team2:
            form_team2 = get_team_form(team2)
        if team1 and team2:
            if selected_type == 'fc':
                prediction_xg = predict_match_fc_xg(date, team1, team2)
                prediction_rf = predict_match_fc_rf(date, team1, team2)
                prediction_lr1 = predict_match_fc_lr(date, team1, team2)
                prediction_lr2 = predict_match_score_fc_lr(date, team1, team2)
                prediction_lr = prediction_lr1 | prediction_lr2
                accuracyXG = "{:.2f}".format(accuracyXG_FA)
                accuracyRF = "{:.2f}".format(accuracyRF_FA)
                accuracyLR = "{:.2f}".format(accuracyLR_FA)
            elif selected_type == 'pl':
                prediction_xg1 = predict_match_premier_league_xg(date, team1, team2)
                prediction_xg2 = predict_match_score_premier_league_xg(date, team1, team2)
                prediction_xg = prediction_xg1 | prediction_xg2
                prediction_rf1 = predict_match_premier_league_rf(date, team1, team2)
                prediction_rf2 = predict_match_score_premier_league_rf(date, team1, team2)
                prediction_rf = prediction_rf1 | prediction_rf2
                accuracyXG = "{:.2f}".format(accuracyXG_PL)
                accuracyRF = "{:.2f}".format(accuracyRF_PL)
                accuracyLR = "{:.2f}".format(accuracyLR_PL)
            return render_template('starting_page.html',
                                   accuracyXG=accuracyXG,
                                   accuracyRF=accuracyRF,
                                   accuracyLR=accuracyLR,
                                   date=date,
                                   team_list=team_list,
                                   form_team1=form_team1,
                                   form_team2=form_team2,
                                   selected_team1=team1,
                                   selected_team2=team2,
                                   selected_type=selected_type,
                                   prediction_xg=prediction_xg,
                                   prediction_rf=prediction_rf,
                                   prediction_lr=prediction_lr,
                                   is_prediction=True,
                                   )

    return render_template('starting_page.html',
                           accuracyXG_PL=formattedXG_PL,
                           accuracyRF_PL=formattedRF_PL,
                           accuracyLR_PL=formattedLR_PL,
                           date=date,
                           team_list=team_list,
                           form_team1=form_team1,
                           form_team2=form_team2,
                           selected_team1=team1,
                           selected_team2=team2,
                           selected_type=selected_type,
                           )



# NERD STATS VIEWS
@app.route('/statsfornerds')
def stats_for_nerds():
    #variables for accuracy graph
    iterasi = ['1', '2', '3']

    # Convert convert string to float soalnya aku banyak gaya pake formatting
    grafik_XG = [float(formattedXG_FA_first), float(formattedXG_FA_second), float(formattedXG_FA)]
    grafik_RF = [float(formattedRF_FA_first), float(formattedRF_FA_second), float(formattedRF_FA)]
    grafik_LR = [float(formattedLR_FA_first), float(formattedLR_FA_second), float(formattedLR_FA)]

    # offset buat nampilin graph secara parallel (i spent about 45 minutes thinking of how to implement this. plshelp.)
    offset = 0.02  # offset value (jujur aja aku ngawur tpi gpp)
    grafik_RF_offset = [x + offset for x in grafik_RF]
    grafik_LR_offset = [x + 2 * offset for x in grafik_LR]

    # set figure plot and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(iterasi, grafik_XG, label='XGBoost Accuracy', marker='o', linestyle='-', linewidth=2, color='blue')
    ax.plot(iterasi, grafik_RF_offset, label='Random Forest Accuracy', marker='o', linestyle='-',
            linewidth=2, color='green')
    ax.plot(iterasi, grafik_LR_offset, label='Logistic Regression Accuracy', marker='o', linestyle='-',
            linewidth=2, color='red')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Model Accuracy Over Iterations', fontsize=14)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.7)

    # Save plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url_perkembangan = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)
    return render_template('stats.html',
                           accuracyXG_FAcup=formattedXG_FA, results_XG_FA=results_XG_FA,
                           accuracyRF_FAcup=formattedRF_FA, results_RF_FA=results_RF_FA,
                           accuracyLR_FAcup=formattedLR_FA, results_LR_FA=results_LR_FA,
                           accuracyXG_PL=formattedXG_PL,
                           accuracyRF_PL=formattedRF_PL,
                           accuracyLR_PL=formattedLR_PL,
                           plot_url_perkembangan=plot_url_perkembangan, confusion_plotXG_FA=confusion_plotXG_FA, confusion_plotRF_FA=confusion_plotRF_FA, confusion_plotLR_FA=confusion_plotLR_FA,
                           metrics_data_XG=metrics_data, division_gap_XG=division_gap, plot_dataXG_FA=plot_dataXG_FA, cmXG_FA=str(cmXG_FA),
                           metrics_data_RF=metrics_data_RF_FA, division_gap_RF_FA=division_gap_RF_FA, plot_dataRF_FA=plot_dataRF_FA, cmRF_FA=str(cmRF_FA),
                           metrics_data_LR=metrics_data_LR_FA, division_gap_LR_FA=division_gap_LR_FA, plot_dataLR_FA=plot_dataLR_FA, cmLR_FA=str(cmLR_FA),
                           )

if __name__ == '__main__':
    app.run(debug=True)