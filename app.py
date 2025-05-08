from flask import Flask, render_template, render_template_string, request, send_file, jsonify 
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Використовуємо non-GUI backend
import matplotlib.pyplot as plt
import io
import base64
import csv
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import Callback

# Створення Flask додатку
app = Flask(__name__)

# Глобальна змінна для зберігання результату
plot_url = None

# === Прогрес статус ===
progress_status = {"progress": 0, "status": "Очікування...", "plot_url": None}

# === Callback ===
class TrainingProgressCallback(Callback):
    def on_train_begin(self, logs=None):
        progress_status.update({"progress": 0, "status": "Початок тренування", "loss": None})

    def on_epoch_end(self, epoch, logs=None):
        total_epochs = self.params['epochs']
        progress = int(((epoch + 1) / total_epochs) * 100)
        progress_status.update({
            "progress": progress,
            "status": f"Епоха {epoch + 1}/{total_epochs}",
            "loss": logs.get("loss")
        })

    def on_train_end(self, logs=None):
        progress_status.update({"progress": 100, "status": "Тренування завершено"})




mean_errors = {
                "Loss" : 0,
                "MAE"  : 0,
                "MSE"  : 0,
                "RMSE" : 0
              }

# Початкові параметри за замовчуванням
DEFAULT_PARAMS = {
    "N": 100000,
    "I0": 1,
    "R0": 0,
    "V0": 0,
    "beta": 0.3,
    "gamma": 0.1,
    "sigma": 0.1,
    "lambda_jump": 0.07,
    "v_rate": 0.25,
    "reinf_rate": 0.005,
    "reinf_factor": 0.5,
    "t_max": 100,
    "dt": 1,
    "vaccine_start": 30,
    "use_neural_net": 0
}

RESULTS = []

# Функція для обробки моделювання
def run_simulation_async(**params):
    global plot_url
    plot_url = run_simulation(**params)  # Викликаємо основну функцію моделювання
    return plot_url

# Використовуємо ThreadPoolExecutor для обробки завдання
executor = ThreadPoolExecutor(2)

def run_simulation(N, I0, R0, V0, beta, gamma, sigma, lambda_jump,
                   v_rate, reinf_rate, reinf_factor, t_max, dt,
                   vaccine_start, use_neural_net):

    global RESULTS
    global mean_errors
    
    mean_errors = {
                "Loss" : 0,
                "MAE"  : 0,
                "MSE"  : 0,
                "RMSE" : 0
              }
    
    S0 = N - I0 - R0 - V0
    time = np.arange(0, t_max, dt)
    S = np.zeros(t_max, dtype=int)
    I = np.zeros(t_max, dtype=int)
    R = np.zeros(t_max, dtype=int)
    V = np.zeros(t_max, dtype=int)

    S[0], I[0], R[0], V[0] = S0, I0, R0, V0

    for t in range(1, t_max):
        dW_S = np.random.normal(0, np.sqrt(dt))
        dW_I = np.random.normal(0, np.sqrt(dt))
        dW_R = np.random.normal(0, np.sqrt(dt))

        poisson_jump = np.random.poisson(lambda_jump * dt)

        dV = 0
        if t >= vaccine_start:
            dV = v_rate * S[t-1] * dt

        dS = -beta * S[t-1] * I[t-1] * dt + sigma * S[t-1] * dW_S - dV + reinf_rate * R[t-1] * dt
        dI = (beta * S[t-1] * I[t-1] - gamma * I[t-1]) * dt + sigma * I[t-1] * dW_I + poisson_jump + reinf_rate * R[t-1] * dt * reinf_factor
        dR = gamma * I[t-1] * dt + sigma * I[t-1] * dW_R - reinf_rate * R[t-1] * dt * reinf_factor

        s_new = max(0, round(S[t-1] + dS))
        i_new = max(0, round(I[t-1] + dI))
        r_new = max(0, round(R[t-1] + dR))
        v_new = max(0, round(V[t-1] + dV))

        total = s_new + i_new + r_new + v_new
        if total > N:
            excess = total - N
            for group, arr in zip(('S','I','R','V'), (s_new, i_new, r_new, v_new)):
                if arr >= excess:
                    if group == 'S': s_new -= excess
                    elif group == 'I': i_new -= excess
                    elif group == 'R': r_new -= excess
                    elif group == 'V': v_new -= excess
                    break

        S[t], I[t], R[t], V[t] = s_new, i_new, r_new, v_new

    RESULTS = list(zip(time, S, I, R, V))

    

    if use_neural_net:
        # Використовуємо нейромережу для коригування результатів
                
        X, y = prepare_data(S, I, R, V, time)

        model = build_model((X.shape[1], 1))

        # model.fit(X, y, epochs=50, batch_size=32)
        model.fit(X, y, epochs=50, batch_size=32, callbacks=[TrainingProgressCallback()])

        val_loss, val_mae, val_mse = model.evaluate(X, y)
        

        # Обчислення RMSE
        val_rmse = np.sqrt(val_mse)

        mean_errors = {
                       "Loss" : val_loss,
                       "MAE"  : val_mae,
                       "MSE"  : val_mse,
                       "RMSE" : val_rmse
                      }
        
        predictions = model.predict(X)

        # Обчислення помилки (наприклад, абсолютна помилка)
        y = np.squeeze(y)  # Виправляємо форму y до (99, 4)
        errors = np.abs(predictions - y)


        # Оновлення результатів моделювання нейромережею
        S = predictions[:, 0]
        I = predictions[:, 1]
        R = predictions[:, 2]
        V = predictions[:, 3]

        min_length = len(time)
        time = time[:min_length-1]
        S = S[:min_length]
        I = I[:min_length]
        R = R[:min_length]
        V = V[:min_length]

        RESULTS = list(zip(time, S, I, R, V))

    fig, ax = plt.subplots(2,1,figsize=(10, 6))
    plt.subplot(2,1,1)
    ax[0].plot(time, S, label='Сприйнятливі (S)', color='blue')
    ax[0].plot(time, I, label='Інфіковані (I)', color='red')
    ax[0].plot(time, R, label='Одужавші (R)', color='green')
    ax[0].plot(time, V, label='Вакциновані (V)', color='purple')

    #ax.set_ylim(0, N)
    ax[0].set_xlabel('Час')
    ax[0].set_ylabel('К-ть осіб (Classic) / Норм. шкала (Neur) ')
    ax[0].set_title('Стохастична модель поширення інфекції')
    ax[0].legend()
    ax[0].grid(True)

    if use_neural_net:
        
        # Візуалізація помилки
        plt.subplot(2,1,2)
        ax[1].plot(errors[:, 0], label='Помилка для S')
        ax[1].plot(errors[:, 1], label='Помилка для I')
        ax[1].plot(errors[:, 2], label='Помилка для R')
        ax[1].plot(errors[:, 3], label='Помилка для V')

        ax[1].set_title('Помилки в прогнозі (S, I, R, V)')
        ax[1].set_xlabel('Samples')
        ax[1].set_ylabel('Error')
        #ax[1].set_title(f"Validation Loss: {val_loss} /n " +
        #                f"Validation MAE: {val_mae}" +
        #                f"Validation MSE: {val_mse}" +
        #                f"Validation RMSE: {val_rmse}")
        ax[1].legend()
        ax[1].grid(True)
    

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()
    return encoded

def prepare_data(S, I, R, V, time):
    # Нормалізація даних
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_S = scaler.fit_transform(np.array(S).reshape(-1, 1))
    normalized_I = scaler.fit_transform(np.array(I).reshape(-1, 1))
    normalized_R = scaler.fit_transform(np.array(R).reshape(-1, 1))
    normalized_V = scaler.fit_transform(np.array(V).reshape(-1, 1))
    
    # Підготовка X, y для навчання нейромережі
    X, y = [], []
    for t in range(1, len(S)):
        X.append([normalized_S[t-1], normalized_I[t-1], normalized_R[t-1], normalized_V[t-1]])  # 4 характеристики
        y.append([normalized_S[t], normalized_I[t], normalized_R[t], normalized_V[t]])  # 4 характеристики

    X = np.array(X)
    y = np.array(y)

    # Форматування даних для нейромережі
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # форма (num_samples, num_features, 1)

    return X, y

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=4))  # 4 виходи (для S, I, R, V)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])
    return model

@app.route('/', methods=['GET', 'POST'])
def index():
    global plot_url
    params = DEFAULT_PARAMS.copy()
    
    use_neural_net = False
    if request.method == 'POST':

        for key, val in params.items():
            val = request.form.get(key, str(params[key]))
            params[key] = float(val) if '.' in val or 'e' in val.lower() else int(val)

        if params['use_neural_net'] == 0:
            use_neural_net = False
        else:
            use_neural_net = True
    
        
        # Запускаємо моделювання у фоновому потоці
        future = executor.submit(run_simulation_async, **params)
        
        # Отримуємо результат після завершення виконання
        plot_url = future.result()

#    return render_template_string(TEMPLATE, params=params, plot_url=plot_url, use_neural_net=use_neural_net, mean_errors=mean_errors)
    return render_template("index.html",
                           params=params,
                           plot_url=plot_url,
                           mean_errors=mean_errors)

@app.route('/download-csv')
def download_csv():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Time", "S", "I", "R", "V"])
    for row in RESULTS:
        writer.writerow(row)
    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode('utf-8')), mimetype="text/csv", as_attachment=True, download_name="simulation_results.csv")

@app.route('/download-pdf')
def download_pdf():
    from matplotlib.backends.backend_pdf import PdfPages

    if not RESULTS:
        return "Немає даних для експорту. Запустіть модель спочатку.", 400
    
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig, ax = plt.subplots(figsize=(10, 6))
        time, S, I, R, V = zip(*RESULTS)
        ax.plot(time, S, label='Сприйнятливі (S)', color='blue')
        ax.plot(time, I, label='Інфіковані (I)', color='red')
        ax.plot(time, R, label='Одужавші (R)', color='green')
        ax.plot(time, V, label='Вакциновані (V)', color='purple')
        ax.set_ylim(0, max(S[0]+I[0]+R[0]+V[0], max(S)))
        ax.set_xlabel('Час')
        ax.set_ylabel('К-ть осіб (class) / Норм. шкала (neur)')
        ax.set_title('Стохастична модель поширення інфекції')
        ax.legend()
        ax.grid(True)
        pdf.savefig(fig)
        plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='application/pdf', as_attachment=True, download_name='simulation_results.pdf')


# Функція для запуску довготривалих обчислень
def long_running_task(params, update_callback=None):

    import time
    
    for i in range(20):
        time.sleep(0.2)
        if update_callback:
            update_callback(i / 20)
            
    """Функція для запуску обчислень у фоновому потоці"""
    # Запуск моделювання без нейромережі
    params['use_neural_net']=False
    run_simulation(**params)
    print(RESULTS)
    time_no_nn, S_no_nn, I_no_nn, R_no_nn, V_no_nn = zip(*RESULTS)
    
    # Запуск моделювання з нейромережею
    params['use_neural_net']=True
    RESULTS_NN = run_simulation(**params)
    time_nn, S_nn, I_nn, R_nn, V_nn = zip(*RESULTS)
    
    # Якщо час для моделей різний, обрізаємо довший
    min_length = min(len(time_no_nn), len(time_nn))

    time_no_nn = time_no_nn[:min_length]
    S_no_nn = S_no_nn[:min_length]
    I_no_nn = I_no_nn[:min_length]
    R_no_nn = R_no_nn[:min_length]
    V_no_nn = V_no_nn[:min_length]

    time_nn = time_nn[:min_length]
    S_nn = S_nn[:min_length]
    I_nn = I_nn[:min_length]
    R_nn = R_nn[:min_length]
    V_nn = V_nn[:min_length]

    # Генерація порівняльного графіку
    fig, ax = plt.subplots(2,1,figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    ax[0].plot(time_no_nn, S_no_nn, label='Сприйнятливі (S) без нейромережі', color='blue')
    ax[0].plot(time_no_nn, I_no_nn, label='Інфіковані (I) без нейромережі', color='red')
    ax[0].plot(time_no_nn, R_no_nn, label='Одужавші (R) без нейромережі', color='green')
    ax[0].plot(time_no_nn, V_no_nn, label='Вакциновані (V) без нейромережі', color='purple')
    ax[0].set_xlabel('Час')
    ax[0].set_ylabel('Кількість осіб')
    ax[0].set_ylim(0, max(max(S_no_nn), max(S_nn)))
    ax[0].set_title('Порівняння результатів моделювання')
    ax[0].legend()
    ax[0].grid(True)

    plt.subplot(2, 1, 2)
    ax[1].plot(time_nn, S_nn, label='Сприйнятливі (S) з нейромережею', linestyle='--', color='blue')
    ax[1].plot(time_nn, I_nn, label='Інфіковані (I) з нейромережею', linestyle='--', color='red')
    ax[1].plot(time_nn, R_nn, label='Одужавші (R) з нейромережею', linestyle='--', color='green')
    ax[1].plot(time_nn, V_nn, label='Вакциновані (V) з нейромережею', linestyle='--', color='purple')
    ax[1].set_xlabel('Час')
    ax[1].set_ylabel('Нормалізована шкала')
    ax[1].legend()
    ax[1].grid(True)

    
    # Збереження графіка у пам'яті
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    #buf.close()
    #plt.close()

    return encoded

def background_task():
    global plot_url, progress_status
    progress_status["status"] = "Стартує обчислення..."

    # Тут можна вставити update-функцію у long_running_task, якщо є
    def update_progress(p):
        progress_status["progress"] = int(p * 100)
        progress_status["status"] = f"Обчислення... {progress_status['progress']}%"

    # Запускаємо твоє моделювання
    result_base64 = long_running_task(DEFAULT_PARAMS, update_callback=update_progress)
    progress_status["plot_url"] = result_base64
    progress_status["status"] = "Готово!"
    progress_status["progress"] = 100

@app.route('/compare', methods=['GET', 'POST'])
def compare_results():
    # Запуск обчислень у фоновому потоці
    thread = threading.Thread(target=background_task)
    thread.start()

    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Порівняння результатів</title>
        <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
        <style>
            .progress-bar {
                width: 0%;
                transition: width 0.5s;
            }
        </style>
    </head>
    <body>
    <div class="container mt-5">
        <h2 class="text-center">Порівняння результатів</h2>

        <div class="my-3">
            <div class="progress">
                <div id="progress-bar" class="progress-bar bg-success" role="progressbar">0%</div>
            </div>
            <p id="status-text" class="text-center mt-2">Очікування...</p>
        </div>

        <div id="plot-container" class="text-center mt-4">
            <img id="plot-img" class="img-fluid border rounded" style="max-height: 400px;">
        </div>
    </div>

    <script>
        function updateProgress() {
            fetch('/progress')
                .then(response => response.json())
                .then(data => {
                    const bar = document.getElementById('progress-bar');
                    const text = document.getElementById('status-text');
                    const img = document.getElementById('plot-img');

                    bar.style.width = data.progress + '%';
                    bar.innerText = data.progress + '%';
                    text.innerText = data.status;

                    if (data.progress >= 100 && data.plot_url) {
                        // Перевіряємо правильність значення plot_url
                        img.src = 'data:image/png;base64,' + data.plot_url;
                    }
                });
        }

        setInterval(updateProgress, 500);
    </script>
    </body>
    </html>
    """)

@app.route('/progress')
def progress():
    return jsonify(progress_status)

@app.route('/get_plot')
def get_plot():
    global plot_url
    print("DEBUG get_plot: Length of plot_url:", len(plot_url))
    return jsonify({"plot_url": plot_url})


@app.route('/start-training', methods=['POST'])
def start_training():
    threading.Thread(target=train_model).start()
    return jsonify({"message": "Тренування розпочато"})

@app.route('/training-progress')
def training_progress():
    return jsonify(progress_status)








if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
