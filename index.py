from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def hello():
    if request.method == 'POST':
        gender = request.form['gender']
        age = request.form['age']
        sport = request.form['sport']
        local_inj = request.form['local_inj']
        region_inj = request.form['region_inj']
        inj_type = request.form['inj_type']
        
        # Exemplo de exibição das escolhas no formato de string
        #result = f"Gênero: {gender}<br>Idade: {age}<br>Desporto: {sport}<br>Local da Lesão: {local_inj}<br>Região Corporal da Lesão: {region_inj}<br>Tipo de Lesão: {inj_type}"
        form_data = [gender, age, sport, local_inj, region_inj, inj_type]
        # Carregar o modelo treinado e outras configurações
        model_xgb = xgb.Booster()
        model_xgb.load_model("modelXGBOOST14jun23.pkl")

        # Definir as variáveis necessárias
        col_original_name = ["gender", "age", "sport", "local_inj", "region_inj", "inj_type"]

        # Ler a lista do arquivo

        column_names = ['gender_female', 'gender_male',
        'age_0-4 years',
        'age_10-14 years',
        'age_15-19 years',
        'age_5-9 years',
        'sport_Aerobics',
        'sport_AirSports-NotMotorised',
        'sport_Badminton',
        'sport_Baseball/Softball',
        'sport_Basketball',
        'sport_Climbing/Mountaineering/Caving',
        'sport_CombatSports',
        'sport_Cricket',
        'sport_Cycling',
        'sport_Football',
        'sport_Golf',
        'sport_Gymnastics(NotTrampoline)',
        'sport_Hiking',
        'sport_Hockey',
        'sport_Horse-Riding',
        'sport_Ice-Skating',
        'sport_Motorsport',
        'sport_Netball',
        'sport_OtherSpecifiedSport',
        'sport_Rollerblades/Skates',
        'sport_Rugby',
        'sport_RunningJogging',
        'sport_SCUBADiving(Recreational)',
        'sport_Skateboarding',
        'sport_Skiing',
        'sport_Snowboarding',
        'sport_Squash',
        'sport_Swimming',
        'sport_Tennis',
        'sport_Trampoline',
        'sport_Watersports-Motorised',
        'sport_Watersports-Non-Motorised',
        'sport_WatersportsNon-Motorised',
        'sport_Weightlifting/StrengthBuilding',
        'local_inj_Athletics_and_Sports_Area',
        'local_inj_Countryside_Beach_Sea',
        'local_inj_Farm',
        'local_inj_Home',
        'local_inj_Other_location',
        'local_inj_Public_Recreational_Area',
        'local_inj_Road_Street_or_Motorway',
        'local_inj_School_Educational_Area',
        'region_inj_Abd_spi_tho_pel',
        'region_inj_Head',
        'region_inj_Lower_limb',
        'region_inj_Site_unspecified',
        'region_inj_Upper_Limb',
        'inj_type_Blow_From_Object',
        'inj_type_Blunt_Force/Pushed',
        'inj_type_Crushing_Injury',
        'inj_type_CuttingPiercing_by_Other_Sharp_Object',
        'inj_type_Fall/Slip/Trip-HIGHoneMetre',
        'inj_type_Fall/Slip/Trip-LOWoneMetre',
        'inj_type_Other',
        'inj_type_Physical_OverExertion/OverExtention',
        'inj_type_Punched/Kicked']

        labels = ["VeryLowRiskOfFracture", "HighRiskOfFracture", "ProbableRiskOfFracture" ]
        labels= pd.DataFrame(labels, columns=["labels"])

        # Rota para receber os dados do formulário e enviar a previsão

        # Obter os dados do formulário
        #form_data = ["male", "5-9 years", "Football", "Athletics_and_Sports_Area", "Upper_Limb", "Fall/Slip/Trip-LOWoneMetre"] #request.get_json()

        # Converter os valores categóricos em valores numéricos
        prevision_call = form_data#[form_data.get(col) for col in col_original_name]
        combined_prevision_call = [x + "_" + y for x, y in zip(col_original_name, prevision_call)]

        new_data_encoded = pd.DataFrame(columns=column_names)

        for col in new_data_encoded.columns:
            if col in combined_prevision_call:
                new_data_encoded.loc[0, col] = 1
            else:
                new_data_encoded.loc[0, col] = 0

        new_data_encoded = new_data_encoded.astype('bool')

        # Converter os dados de entrada em uma matriz DMatrix
        dnew = xgb.DMatrix(data=new_data_encoded, feature_names=column_names)

        # Fazer a previsão
        predictions = model_xgb.predict(dnew)
        predicted_labels = np.argmax(predictions, axis=1)  # Obter as classes previstas

        # Converter as classes previstas para os rótulos originais
        le = LabelEncoder()
        le.fit(labels)  # Substitua y_train_resampled pelos seus rótulos originais
        predicted_classes = le.inverse_transform(predicted_labels)

        # Retornar a resposta como JSON
        response = {'prediction': predicted_classes[0]}

        
        return response
    else:
        return render_template("index.html")
if __name__ == "__main__":
    app.run()