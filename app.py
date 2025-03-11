import firebase_admin
from firebase_admin import credentials, auth, storage
from firebase_admin import firestore
import joblib
import pandas as pd
from flask import Flask, render_template, request, session, redirect
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import datetime
from sklearn.metrics import r2_score
from flask_mail import Mail, Message
import pyrebase

app = Flask(__name__)
config = {
    'apiKey': "AIzaSyCi9Fm7aqJ8TdMgRt3mobzr_iTDYtegMrY",
    'authDomain': "forescope-c6869.firebaseapp.com",
    'databaseURL': "https://forescope-c6869-default-rtdb.firebaseio.com",
    'projectId': "forescope-c6869",
    'storageBucket': "forescope-c6869.appspot.com",
    'messagingSenderId': "916323027903",
    'appId': "1:916323027903:web:c546b240fe8bba841f6f0a",
    'measurementId': "G-EPSXX3JRLT"
}

firebase = pyrebase.initialize_app(config)
authenticate = firebase.auth()
fstorage = firebase.storage()
app.secret_key = 'secret'

app.config['SECRET_KEY'] = "tsfyguaistyatuis589566875623568956"
app.config['MAIL_SERVER'] = "smtp.googlemail.com"
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = "teamforescope@gmail.com"
app.config['MAIL_PASSWORD'] = "cnrrmtjppkggauyd"
mail = Mail(app)

cred = credentials.Certificate('forescope_key.json')
firebase_admin.initialize_app(cred, {'storageBucket': 'forescope-c6869.appspot.com'})
db = firestore.client()

alerta_ref = db.collection('alertas').document('alerta_principal')
# alerta = alerta_ref.get()


nuevo_modelo_entrenado = joblib.load('models/modelRF.pkl')
nuevo_modelo_entrenado_z1 = joblib.load('models/modelRF_z1.pkl')
nuevo_modelo_entrenado_z2 = joblib.load('models/modelRF_z2.pkl')
nuevo_modelo_entrenado_z3 = joblib.load('models/modelRF_z3.pkl')
nuevo_modelo_entrenado_z4 = joblib.load('models/modelRF_z4.pkl')
nuevo_modelo_entrenado_z5 = joblib.load('models/modelRF_z5.pkl')
nuevo_modelo_entrenado_z6 = joblib.load('models/modelRF_z6.pkl')


@app.route('/')
def entrance():  # put application's code here
    return redirect('/login')


def datos_usuario(user):
    user_id = session[user]
    usuario_ref = db.collection('usuarios').document(user_id)
    usuario_doc = usuario_ref.get()
    if usuario_doc.exists:
        nombre_usuario = usuario_doc.get("nombre_usuario")
    user_data = auth.get_user(user_id)
    return usuario_doc


@app.route('/dashboard')
def dashboard():
    if 'user' in session:
        user_id = session['user']
        photo_url = fstorage.child(f'user_profile_photos/{user_id}.jpg').get_url(None)
        nombre = datos_usuario('user').get("nombre_usuario")

        def generate_predictions_with_special_months(model, initial_date):
            # Crear un DataFrame con 10 días siguientes a la fecha inicial
            dates = pd.date_range(start=initial_date, periods=10, freq='D')

            data = {'dayofweek': dates.dayofweek, 'Mes': dates.month, 'Anio': dates.year, 'quarter': dates.quarter,
                    'week_of_year': dates.isocalendar().week,
                    'day_of_year': dates.day_of_year,
                    'is_leap': dates.is_leap_year, 'cantidad_dias_mes': dates.daysinmonth}
            future_df = pd.DataFrame(data)
            predictions = model.predict(future_df)
            fechas_formateadas = [dates.strftime('%d %b') for dates in dates]

            results = pd.DataFrame({'Date': fechas_formateadas, 'demanda pronosticada': predictions})
            return results

        initial_date = pd.to_datetime(datetime.datetime.now())
        new_data = [
            [initial_date.dayofweek, initial_date.month, initial_date.year, initial_date.quarter,
             initial_date.weekofyear, initial_date.day_of_year, initial_date.is_leap_year, initial_date.daysinmonth]]

        demanda_rf = generate_predictions_with_special_months(nuevo_modelo_entrenado, initial_date)

        predicted_demand = nuevo_modelo_entrenado.predict(new_data)
        new_cantidad_total = predicted_demand[0]

        new_data_zonas = [
            [new_cantidad_total, initial_date.dayofweek, initial_date.month, initial_date.year, initial_date.quarter,
             initial_date.weekofyear,
             initial_date.day_of_year, initial_date.is_leap_year, initial_date.daysinmonth]]

        current_date = datetime.datetime.now()

        demanda_z1 = int(nuevo_modelo_entrenado_z1.predict(new_data_zonas)[0])
        demanda_z2 = int(nuevo_modelo_entrenado_z2.predict(new_data_zonas)[0])
        demanda_z3 = int(nuevo_modelo_entrenado_z3.predict(new_data_zonas)[0])
        demanda_z4 = int(nuevo_modelo_entrenado_z4.predict(new_data_zonas)[0])
        demanda_z5 = int(nuevo_modelo_entrenado_z5.predict(new_data_zonas)[0])
        demanda_z6 = int(nuevo_modelo_entrenado_z6.predict(new_data_zonas)[0])

        prediccion_hoy = int(round(nuevo_modelo_entrenado.predict(new_data)[0]))
        fechas_10 = demanda_rf['Date'].tolist()
        demanda_10 = demanda_rf['demanda pronosticada'].tolist()

        # doc_ref = db.collection('alertas').document('alerta_principal')
        # doc = doc_ref.get()

        alerta = alerta_ref.get()
        if alerta.exists:
            treshold = alerta.to_dict().get('treshold')
            if prediccion_hoy > treshold:
                send_alert_email()
                eye_slash = ' '
                ds_msj_main_mail = 'Existe pronóstico por encima de lo esperado'
                ds_msj_sub_mail = 'Se notificó por email al responsable'
                fondo_ojo = '#f0ad4e'
            else:
                eye_slash = '-slash'
                ds_msj_main_mail = 'Demanda dentro de lo esperado'
                ds_msj_sub_mail = 'No se han tomado medidas'
                fondo_ojo = '#9b9898'
        else:
            eye_slash = '-slash'
            ds_msj_main_mail = 'No hay alertas registradas'
            ds_msj_sub_mail = ' '
            fondo_ojo = '#9b9898'

        if alerta.exists:
            return render_template('dashboard.html', demanda_rf=demanda_rf, fecha_hoy=current_date,
                                   demanda_hoy=prediccion_hoy, fechas_10=fechas_10, demanda_10dias=demanda_10,
                                   active_dashboard='active', color='#cfe5dc', ds_msj_main='Alerta configurada',
                                   ds_msj_sub='se te notificará', eye_slash=eye_slash,
                                   ds_msj_main_mail=ds_msj_main_mail,
                                   ds_msj_sub_mail=ds_msj_sub_mail, fondo_ojo=fondo_ojo, demanda_z1=demanda_z1,
                                   demanda_z2=demanda_z2, demanda_z3=demanda_z3, demanda_z4=demanda_z4,
                                   demanda_z5=demanda_z5, demanda_z6=demanda_z6, usuario=nombre, user_profile_photo_url=photo_url)
        else:
            return render_template('dashboard.html', demanda_rf=demanda_rf, fecha_hoy=current_date,
                                   demanda_hoy=prediccion_hoy, fechas_10=fechas_10, demanda_10dias=demanda_10,
                                   active_dashboard='active', color='#6c757d', ds_msj_main='Alerta no configurada',
                                   ds_msj_sub=' ', eye_slash=eye_slash, ds_msj_main_mail=ds_msj_main_mail,
                                   ds_msj_sub_mail=ds_msj_sub_mail, fondo_ojo=fondo_ojo, demanda_z1=demanda_z1,
                                   demanda_z2=demanda_z2, demanda_z3=demanda_z3, demanda_z4=demanda_z4,
                                   demanda_z5=demanda_z5, demanda_z6=demanda_z6, usuario=nombre, user_profile_photo_url=photo_url)
    else:
        return redirect('/login')


def send_alert_email():
    # doc_ref = db.collection('alertas').document('alerta_principal')
    # doc = doc_ref.get()
    alerta = alerta_ref.get()
    if alerta.exists:
        email = alerta.to_dict().get('correo')
        msg_title = "This is a test email"
        sender = "noreply@app.com"
        msg = Message(msg_title, sender=sender, recipients=[email])
        msg_body = "This is the email body"
        msg.body = ""
        data = {
            'app_name': "Forescope",
            'title': msg_title,
            'body': msg_body,
        }
        msg.html = render_template("email.html", data=data)

        mail.send(msg)


@app.route('/entrenar')
def entrenar():
    if 'user' in session:
        user_id = session['user']
        photo_url = fstorage.child(f'user_profile_photos/{user_id}.jpg').get_url(None)
        nombre = datos_usuario('user').get("nombre_usuario")
        return render_template('entrenar.html', alert_display='none', display_confirmar_entrenar='none',
                               active_entrenar='active', usuario=nombre, user_profile_photo_url=photo_url)
    else:
        return redirect('/login')


@app.route('/upload', methods=['POST'])
def upload():
    if 'user' in session:
        user_id = session['user']
        photo_url = fstorage.child(f'user_profile_photos/{user_id}.jpg').get_url(None)
        nombre = datos_usuario('user').get("nombre_usuario")
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']

        if file.filename == '':
            return render_template('entrenar.html', alert_display='block', display_confirmar_entrenar='none',
                                   active_entrenar='active', usuario=nombre, user_profile_photo_url=photo_url)

        # Guarda el archivo en el sistema
        uploaded_file_path = file.filename

        file.save(uploaded_file_path)

        # Procesa el archivo CSV
        data = pd.read_csv(uploaded_file_path)
        data_z1 = pd.read_csv(uploaded_file_path)
        data['fecha'] = pd.to_datetime(data['fecha'])
        data['dayofweek'] = data['fecha'].dt.dayofweek
        data['Mes'] = data['fecha'].dt.month
        data['Anio'] = data['fecha'].dt.year
        data['quarter'] = data['fecha'].dt.quarter
        data['day_of_month'] = data['fecha'].dt.day
        data['week_of_year'] = data['fecha'].dt.isocalendar().week
        data['day_of_year'] = data['fecha'].dt.day_of_year
        data['is_leap'] = data['fecha'].dt.is_leap_year
        data['cantidad_dias_mes'] = data['fecha'].dt.daysinmonth
        # data['inicio_mes'] = data['fecha'].dt.is_month_start
        # data['fin_mes'] = data['fecha'].dt.is_month_end

        # Dividir en características (X) y etiquetas (y)
        X = data[['dayofweek', 'Mes', 'Anio', 'quarter', 'week_of_year', 'day_of_year', 'is_leap',
                  'cantidad_dias_mes']]
        X_z1 = data[
            ['cantidad.incidente', 'dayofweek', 'Mes', 'Anio', 'quarter', 'week_of_year', 'day_of_year',
             'is_leap', 'cantidad_dias_mes']]
        X_z2 = data[
            ['cantidad.incidente', 'dayofweek', 'Mes', 'Anio', 'quarter', 'week_of_year', 'day_of_year',
             'is_leap', 'cantidad_dias_mes']]
        X_z3 = data[
            ['cantidad.incidente', 'dayofweek', 'Mes', 'Anio', 'quarter', 'week_of_year', 'day_of_year',
             'is_leap', 'cantidad_dias_mes']]
        X_z4 = data[
            ['cantidad.incidente', 'dayofweek', 'Mes', 'Anio', 'quarter', 'week_of_year', 'day_of_year',
             'is_leap', 'cantidad_dias_mes']]
        X_z5 = data[
            ['cantidad.incidente', 'dayofweek', 'Mes', 'Anio', 'quarter', 'week_of_year', 'day_of_year',
             'is_leap', 'cantidad_dias_mes']]
        X_z6 = data[
            ['cantidad.incidente', 'dayofweek', 'Mes', 'Anio', 'quarter', 'week_of_year', 'day_of_year',
             'is_leap', 'cantidad_dias_mes']]
        y = data['cantidad.incidente']
        y_z1 = data['cantidad_z1']
        y_z2 = data['cantidad_z2']
        y_z3 = data['cantidad_z3']
        y_z4 = data['cantidad_z4']
        y_z5 = data['cantidad_z5']
        y_z6 = data['cantidad_z6']

        # Dividir en conjunto de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
        X_z1_train, X_z1_test, y_z1_train, y_z1_test = train_test_split(X_z1, y_z1, test_size=0.35, random_state=42)
        X_z2_train, X_z2_test, y_z2_train, y_z2_test = train_test_split(X_z2, y_z2, test_size=0.35, random_state=42)
        X_z3_train, X_z3_test, y_z3_train, y_z3_test = train_test_split(X_z3, y_z3, test_size=0.35, random_state=42)
        X_z4_train, X_z4_test, y_z4_train, y_z4_test = train_test_split(X_z4, y_z4, test_size=0.35, random_state=42)
        X_z5_train, X_z5_test, y_z5_train, y_z5_test = train_test_split(X_z5, y_z5, test_size=0.35, random_state=42)
        X_z6_train, X_z6_test, y_z6_train, y_z6_test = train_test_split(X_z6, y_z6, test_size=0.35, random_state=42)

        global nuevo_modelo_entrenado
        global nuevo_modelo_entrenado_z1
        global nuevo_modelo_entrenado_z2
        global nuevo_modelo_entrenado_z3
        global nuevo_modelo_entrenado_z4
        global nuevo_modelo_entrenado_z5
        global nuevo_modelo_entrenado_z6

        nuevo_modelo_entrenado = RandomForestRegressor(n_estimators=150, random_state=128, max_depth=None,
                                                       min_samples_split=10, min_samples_leaf=4)
        nuevo_modelo_entrenado.fit(X_train, y_train)
        y_pred = nuevo_modelo_entrenado.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        nuevo_modelo_entrenado_z1 = RandomForestRegressor(n_estimators=150, random_state=128, max_depth=None,
                                                          min_samples_split=10, min_samples_leaf=4)
        nuevo_modelo_entrenado_z1.fit(X_z1_train, y_z1_train)
        y_z1_pred = nuevo_modelo_entrenado_z1.predict(X_z1_test)
        r2_z1 = r2_score(y_z1_test, y_z1_pred)

        nuevo_modelo_entrenado_z2 = RandomForestRegressor(n_estimators=150, random_state=128, max_depth=None,
                                                          min_samples_split=10, min_samples_leaf=4)
        nuevo_modelo_entrenado_z2.fit(X_z2_train, y_z2_train)

        nuevo_modelo_entrenado_z3 = RandomForestRegressor(n_estimators=150, random_state=128, max_depth=None,
                                                          min_samples_split=10, min_samples_leaf=4)
        nuevo_modelo_entrenado_z3.fit(X_z3_train, y_z3_train)

        nuevo_modelo_entrenado_z4 = RandomForestRegressor(n_estimators=150, random_state=128, max_depth=None,
                                                          min_samples_split=10, min_samples_leaf=4)
        nuevo_modelo_entrenado_z4.fit(X_z4_train, y_z4_train)

        nuevo_modelo_entrenado_z5 = RandomForestRegressor(n_estimators=150, random_state=128, max_depth=None,
                                                          min_samples_split=10, min_samples_leaf=4)
        nuevo_modelo_entrenado_z5.fit(X_z5_train, y_z5_train)

        nuevo_modelo_entrenado_z6 = RandomForestRegressor(n_estimators=150, random_state=128, max_depth=None,
                                                          min_samples_split=10, min_samples_leaf=4)
        nuevo_modelo_entrenado_z6.fit(X_z6_train, y_z6_train)

        # import pickle
        # pickle.dump(model, open('modelRF.pkl', 'wb'))
        return render_template('entrenar.html', r2='La precisión del modelo entrenado es de: {}'.format(r2),
                               alert_display='none', display_confirmar_entrenar='block', active_entrenar='active',
                               usuario=nombre, user_profile_photo_url=photo_url)
    else:
        return redirect('/login')


@app.route('/save-model', methods=['POST'])
def save_model():
    import pickle
    pickle.dump(nuevo_modelo_entrenado, open('models/modelRF.pkl', 'wb'))
    pickle.dump(nuevo_modelo_entrenado_z1, open('models/modelRF_z1.pkl', 'wb'))
    pickle.dump(nuevo_modelo_entrenado_z2, open('models/modelRF_z2.pkl', 'wb'))
    pickle.dump(nuevo_modelo_entrenado_z3, open('models/modelRF_z3.pkl', 'wb'))
    pickle.dump(nuevo_modelo_entrenado_z4, open('models/modelRF_z4.pkl', 'wb'))
    pickle.dump(nuevo_modelo_entrenado_z5, open('models/modelRF_z5.pkl', 'wb'))
    pickle.dump(nuevo_modelo_entrenado_z6, open('models/modelRF_z6.pkl', 'wb'))

    return "Modelo actualizado y guardado"


@app.route('/pronostico_z1')
def pronostico_z1():
    return render_template('pronostico_zona_z1.html', methods=['GET'], display_alert_z1='none')


@app.route('/prediccion_z1', methods=['POST'])
def prediccion_z1():
    # fecha = int(request.form['fecha'])
    fecha = request.form['fecha']
    new_date = pd.to_datetime(fecha)

    new_data = [[new_date.dayofweek, new_date.month, new_date.year, new_date.quarter, new_date.weekofyear,
                 new_date.day_of_year, new_date.is_leap_year, new_date.daysinmonth]]
    predicted_demand = nuevo_modelo_entrenado.predict(new_data)
    new_cantidad_total = predicted_demand[0]

    new_data = [[new_cantidad_total, new_date.dayofweek, new_date.month, new_date.year, new_date.quarter,
                 new_date.weekofyear,
                 new_date.day_of_year, new_date.is_leap_year, new_date.daysinmonth]]
    demanda_rf_z1 = nuevo_modelo_entrenado_z1.predict(new_data)
    # demanda_rf_z1 = float(demanda_rf_z1[0])
    # demanda_rf_z1 = round(demanda_rf_z1, 2)

    return render_template('pronostico_zona_z1.html', demanda_zona_z1=demanda_rf_z1[0], display_alert_z1='block')


@app.route('/pronostico_dia')
def pronostico_dia():
    if 'user' in session:
        user_id = session['user']
        photo_url = fstorage.child(f'user_profile_photos/{user_id}.jpg').get_url(None)
        nombre = datos_usuario('user').get("nombre_usuario")
        return render_template('pronostico_dia.html', methods=['GET'], display_alert_dia='none',
                               display_caja_dia='none',
                               active_pronostico_dia='active', usuario=nombre, user_profile_photo_url=photo_url)
    else:
        return redirect('/login')


@app.route('/btn_prediccion_dia', methods=['POST'])
def btn_prediccion_dia():
    if 'user' in session:
        user_id = session['user']
        photo_url = fstorage.child(f'user_profile_photos/{user_id}.jpg').get_url(None)
        nombre = datos_usuario('user').get("nombre_usuario")
        fecha = request.form['fecha']
        new_date = pd.to_datetime(fecha)

        new_data = [[new_date.dayofweek, new_date.month, new_date.year, new_date.quarter, new_date.weekofyear,
                     new_date.day_of_year, new_date.is_leap_year, new_date.daysinmonth]]
        predicted_demand = nuevo_modelo_entrenado.predict(new_data)
        new_cantidad_total = predicted_demand[0]

        new_data = [[new_cantidad_total, new_date.dayofweek, new_date.month, new_date.year, new_date.quarter,
                     new_date.weekofyear,
                     new_date.day_of_year, new_date.is_leap_year, new_date.daysinmonth]]

        for i in range(1, 7):
            if i == 1:
                demanda_rf_z1 = nuevo_modelo_entrenado_z1.predict(new_data)
            elif i == 2:
                demanda_rf_z2 = nuevo_modelo_entrenado_z2.predict(new_data)
            elif i == 3:
                demanda_rf_z3 = nuevo_modelo_entrenado_z3.predict(new_data)
            elif i == 4:
                demanda_rf_z4 = nuevo_modelo_entrenado_z4.predict(new_data)
            elif i == 5:
                demanda_rf_z5 = nuevo_modelo_entrenado_z5.predict(new_data)
            elif i == 6:
                demanda_rf_z6 = nuevo_modelo_entrenado_z6.predict(new_data)

        return render_template('pronostico_dia.html', pronostico_dia=int(round(new_cantidad_total)),
                               new_date=new_date.strftime('%d-%m-%Y'), zona_1=demanda_rf_z1[0],
                               zona_2=demanda_rf_z2[0], zona_3=demanda_rf_z3[0], zona_4=demanda_rf_z4[0],
                               zona_5=demanda_rf_z5[0],
                               zona_6=demanda_rf_z6[0], display_alert_dia='block', display_caja_dia='block',
                               active_pronostico_dia='active', usuario=nombre, user_profile_photo_url=photo_url)
    else:
        return redirect('/login')


@app.route('/prediccion_zona_dinamica', methods=['GET', 'POST'])
def prediccion_zona_dinamica():
    if 'user' in session:
        user_id = session['user']
        photo_url = fstorage.child(f'user_profile_photos/{user_id}.jpg').get_url(None)
        nombre = datos_usuario('user').get("nombre_usuario")
        if request.method == 'POST':
            modelo_seleccionado = request.form['modelo']
            fecha_ingresada = request.form['fecha']
            new_date = pd.to_datetime(fecha_ingresada)

            modelo = joblib.load(f"models/modelRF_{modelo_seleccionado}.pkl")

            new_data = [
                [new_date.dayofweek, new_date.month, new_date.year, new_date.quarter, new_date.weekofyear,
                 new_date.day_of_year, new_date.is_leap_year, new_date.daysinmonth]]
            predicted_demand = nuevo_modelo_entrenado.predict(new_data)
            new_cantidad_total = predicted_demand[0]

            new_data = [
                [new_cantidad_total, new_date.dayofweek, new_date.month, new_date.year, new_date.quarter,
                 new_date.weekofyear,
                 new_date.day_of_year, new_date.is_leap_year, new_date.daysinmonth]]
            predicted_demand = modelo.predict(new_data)
            resultados_zonax = predicted_demand[0]

            return render_template('pronostico_zona_dinamica.html', resultados_zonax=resultados_zonax,
                                   display_alert_zona_dinamica='block', modelo_seleccionado=modelo_seleccionado,
                                   usuario=nombre, user_profile_photo_url=photo_url)

        return render_template('pronostico_zona_dinamica.html', display_alert_zona_dinamica='none', usuario=nombre, user_profile_photo_url=photo_url)
    else:
        return redirect('/login')


# registrar alerta original
@app.route('/alerta', methods=['GET', 'POST'])
def alerta():
    if request.method == 'POST':
        correo = request.form.get('correo')
        treshold = int(request.form.get('treshold'))
        mensaje = request.form.get('mensaje')

        alerta_ref.set({'correo': correo, 'treshold': treshold, 'mensaje': mensaje})
        return render_template('nueva_alerta.html', display_alerta='block')

    return render_template('nueva_alerta.html', display_alerta='none')


# fin registrar alerta original

@app.route('/listar_alertas', methods=['GET', 'POST'])
def listar_alertas():
    alerta = alerta_ref.get().to_dict()
    if 'user' in session:
        user_id = session['user']
        photo_url = fstorage.child(f'user_profile_photos/{user_id}.jpg').get_url(None)
        nombre = datos_usuario('user').get("nombre_usuario")

        if request.method == 'POST':
            correo = request.form.get('correo')
            treshold = int(request.form.get('treshold'))
            mensaje = request.form.get('mensaje')

            alerta_ref.set({'correo': correo, 'treshold': treshold, 'mensaje': mensaje})
            alerta = alerta_ref.get().to_dict()
            return render_template('listar_alertas.html', alerta=alerta, display_alerta='block', usuario=nombre, user_profile_photo_url=photo_url)

        alerta = alerta_ref.get().to_dict()
        return render_template('listar_alerta.html', alerta=alerta, display_alerta='none', usuario=nombre, user_profile_photo_url=photo_url)


    else:
        return redirect('/login')


@app.route('/editar_alerta')
def editar_alerta():
    if 'user' in session:
        user_id = session['user']
        photo_url = fstorage.child(f'user_profile_photos/{user_id}.jpg').get_url(None)
        nombre = datos_usuario('user').get("nombre_usuario")
        alerta = alerta_ref.get().to_dict()
        return render_template('editar_alerta.html', alerta=alerta, display_alerta_editar='none', usuario=nombre, user_profile_photo_url=photo_url)
    else:
        return redirect('/login')


@app.route('/guardar', methods=['POST'])
def guardar_cambios():
    if 'user' in session:
        user_id = session['user']
        photo_url = fstorage.child(f'user_profile_photos/{user_id}.jpg').get_url(None)
        nombre = datos_usuario('user').get("nombre_usuario")
        correo = request.form['correo']
        treshold = int(request.form['treshold'])
        mensaje = request.form['mensaje']

        alerta_ref.update({
            'correo': correo,
            'treshold': treshold,
            'mensaje': mensaje
        })
        alerta = alerta_ref.get().to_dict()
        return render_template('editar_alerta.html', alerta=alerta, display_alerta_editar='block', usuario=nombre, user_profile_photo_url=photo_url)
    else:
        return redirect('/login')


@app.route('/nuevo_usuario', methods=['GET', 'POST'])
def nuevo_usuario():
    if request.method == 'POST':
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']
        fullname = request.form['fullname']
        rol_equipo = request.form['rol_equipo']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            error = "Las contraseñas no coinciden."
            return render_template('nuevo_usuario.html', error=error, display_alerta_verde="none")

        try:
            user = auth.create_user(
                email=email,
                password=password
            )
            db = firestore.client()
            user_ref = db.collection('usuarios').document(user.uid)
            user_ref.set({
                'nombre_usuario': username,
                'correo': email,
                'fullname': fullname,
                'rol_equipo': rol_equipo
            })
            # Usuario registrado exitosamente
            return render_template('nuevo_usuario.html', display_alerta_verde="block")
        except Exception as e:
            # Manejo de errores
            return render_template('nuevo_usuario.html', display_alerta_verde="none", error=str(e))

    return render_template('nuevo_usuario.html', display_alerta_verde="none")


@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        try:
            user = authenticate.sign_in_with_email_and_password(email, password)
            session['user'] = user['localId']
            return redirect('/dashboard')
        except Exception as e:
            return render_template('login.html', display_erorr_login="block")
    return render_template('login.html', display_erorr_login="none")


@app.route('/logout')
def logout():
    session.pop('user')
    return redirect('/login')


@app.route('/editar_usuario', methods=['POST', 'GET'])
def editar_usuario():
    if 'user' in session:
        user_id = session['user']
        photo_url = fstorage.child(f'user_profile_photos/{user_id}.jpg').get_url(None)
        nombre = datos_usuario('user').get("nombre_usuario")
        fullnombre = datos_usuario('user').get("fullname")
        rol_equipo = datos_usuario('user').get("rol_equipo")
        correo = datos_usuario('user').get("correo")

        if request.method == 'POST':
            if 'photo' in request.files:
                photo = request.files['photo']
                if photo:
                    # Sube la foto a Firebase Storage
                    bucket = storage.bucket()
                    blob = bucket.blob(
                        f'user_profile_photos/{user_id}.jpg')  # Cambia el nombre y formato de la imagen según tu preferencia
                    blob.upload_from_string(photo.read(), content_type='image/jpeg')

                    # Obtén la URL de descarga de Firebase Storage
                    # photo_url = blob.public_url
                    photo_url = fstorage.child(f'user_profile_photos/{user_id}.jpg').get_url(None)

                    # Guarda la URL de la foto en la base de datos (Firestore, Realtime Database, etc.) para el usuario autenticado
                    # Implementa esta parte según la base de datos que estés usando

                    return render_template('editar_usuario.html', usuario=nombre, fullnombre=fullnombre, rol_equipo=rol_equipo, correo=correo, user_profile_photo_url=photo_url,
                                           navbar_photo_url=photo_url)
        return render_template('editar_usuario.html', usuario=nombre, fullnombre=fullnombre, rol_equipo=rol_equipo,  correo=correo, user_profile_photo_url=photo_url,
                               navbar_photo_url=photo_url)
    else:
        return redirect('/login')

@app.route('/cambios_usuario', methods=['POST'])
def cambios_usuario():
    fullname = request.form['fullname']
    nombre_usuario = request.form['username']
    rol_equipo = request.form['cargo']
    user_id = session['user']
    usuario_ref = db.collection('usuarios').document(session['user'])

    usuario_ref.update({
        'fullname': fullname,
        'nombre_usuario': nombre_usuario,
        'rol_equipo': rol_equipo
    })
    return redirect('/dashboard')

if __name__ == '__main__':
    app.run()
