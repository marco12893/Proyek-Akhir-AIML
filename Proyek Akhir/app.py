from flask import Flask, render_template, request,jsonify,Response
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
import io
from sklearn.preprocessing import MinMaxScaler

#Import method ataupun variable dari file yg diperlukan
from XGBoost_FA_Cup_3 import accuracyFormatted as accuracyXG_FA, df, confidence_df as results_XG_FA, metrics_data
from XGBoost_FA_Cup_2 import accuracyFormatted as accuracyXG_FA_second
from XGBoost_FA_Cup import accuracyFormatted as accuracyXG_FA_first

from Random_Forest_FA_Cup_3 import accuracyFormatted as accuracyRF_FA, confidence_df as results_RF_FA, metrics_data as metrics_data_RF
from Random_Forest_FA_Cup_2 import accuracyFormatted as accuracyRF_FA_second
from Random_Forest_FA_Cup import accuracyFormatted as accuracyRF_FA_first

from Logistic_Regression_FA_Cup_3 import accuracyFormatted as accuracyLR_FA, confidence_df as results_LR_FA, metrics_data as metrics_data_LR
from Logistic_Regression_FA_Cup_2 import accuracyFormatted as accuracyLR_FA_second
from Logistic_Regression_FA_Cup import accuracyFormatted as accuracyLR_FA_first

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
- [SUDAH] tombol stats for nerds di masukin di navbar header
- [SUDAH] grafik yg nunjukin progress kita (3 algoritma dari 3 file) (route yg beda)
- [SUDAH] classification report (route yg beda juga), boleh tunjukin performance by division gap
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

# Ambil team name untuk option bar (masih salah mungkin)
FA_df = df[(df['Type'] == 'FA Cup')]
unique_values = FA_df['Home'].unique()
team_list = unique_values.tolist()


app = Flask(__name__)
# MAIN ROUTE VIEW
@app.route('/')
def home():
    return render_template('starting_page.html',
                           accuracyXG_FAcup=formattedXG_FA, results_XG_FA=results_XG_FA,
                           accuracyRF_FAcup=formattedRF_FA, results_RF_FA=results_RF_FA,
                           accuracyLR_FAcup=formattedLR_FA, results_LR_FA=results_LR_FA,
                           accuracyXG_PL=formattedXG_PL,
                           accuracyRF_PL=formattedRF_PL,
                           accuracyLR_PL=formattedLR_PL,
                           team_list=team_list)



# NERD STATS VIEWS
@app.route('/statsfornerds')
def stats_for_nerds():
    # Variables for accuracy graph
    iterasi = ['1', '2', '3']
    grafik_XG = [formattedXG_FA_first, formattedXG_FA_second, formattedXG_FA]
    grafik_RF = [formattedRF_FA_first, formattedRF_FA_second, formattedRF_FA]
    grafik_LR = [formattedLR_FA_first, formattedLR_FA_second, formattedLR_FA]
    # Standardize accuracy values
    scaler = MinMaxScaler()
    data = np.array([grafik_XG, grafik_RF, grafik_LR]).T  # Transpose to shape (3, 3)
    normalized_data = scaler.fit_transform(data).T  # Normalize and transpose back

    # Assign normalized values
    norm_grafik_XG, norm_grafik_RF, norm_grafik_LR = normalized_data

    # Generate graph
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(iterasi, norm_grafik_XG, label='XGBoost Accuracy', marker='o', linestyle='-', linewidth=2, color='blue')
    ax.plot(iterasi, norm_grafik_RF, label='Random Forest Accuracy', marker='s', linestyle='--', linewidth=2,
            color='green')
    ax.plot(iterasi, norm_grafik_LR, label='Logistic Regression Accuracy', marker='^', linestyle='-.', linewidth=2,
            color='red')

    # Set labels, title, and legend
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Normalized Accuracy', fontsize=12)
    ax.set_title('Model Accuracy Over Iterations (Normalized)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Save plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)
    return render_template('stats.html',
                           plot_url=plot_url,
                           metrics_data_XG=metrics_data,
                           metrics_data_RF=metrics_data_RF,
                           metrics_data_LR=metrics_data_LR,)

if __name__ == '__main__':
    app.run(debug=True)