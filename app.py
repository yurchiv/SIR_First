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


# Створення Flask додатку
app = Flask(__name__)

mean_errors = {
                "Loss" : 0,
                "MAE"  : 0,
                "MSE"  : 0,
                "RMSE" : 0
              }

# Шаблон для веб-сторінки
TEMPLATE = '''
<!doctype html>
<title>Епідеміологічна модель</title>
<style>
  /* Загальний стиль для сторінки */
  body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 0;
  }

  /* Стилі для header */
  header {
    background-color: #4CAF50;
    color: white;
    text-align: center;
    padding: 10px 0;
  }

  /* Стилі для меню */
  nav {
    background-color: #f2f2f2;
    padding: 10px;
    text-align: center;
  }

  nav a {
    margin: 0 15px;
    color: #4CAF50;
    text-decoration: none;
  }

  nav a:hover {
    text-decoration: underline;
  }

  /* Стилі для footer */
  footer {
    background-color: #4CAF50;
    color: white;
    text-align: center;
    padding: 10px 0;
    position: absolute;
    width: 100%;
    bottom: 0;
  }

  /* Система колонок */
  .container {
    display: flex;
    min-height: 80vh; /* Висота основного контенту */
    padding: 20px;
  }

  /* Ліва колонка (параметри) */
  .left-column {
    width: 30%; /* Ширина лівої колонки */
    padding: 20px;
    border-right: 2px solid #ddd;
  }

  /* Права колонка (графік) */
  .right-column {
    width: 70%; /* Ширина правої колонки */
    padding: 20px;
  }

  /* Стилі для форм */
  input[type="number"] {
    margin: 5px 0;
    padding: 5px;
    width: 100%;
  }

  input[type="checkbox"] {
    margin: 10px 0;
  }

  input[type="submit"] {
    background-color: #4CAF50;
    color: white;
    padding: 10px;
    border: none;
    cursor: pointer;
    width: 100%;
  }

  input[type="submit"]:hover {
    background-color: #45a049;
  }

  /* Стилі для зображення графіку */
  .result img {
    max-width: 100%;
    height: auto;
  }

  /* Стиль для результатів */
  .result {
    margin-top: 20px;
  }
</style>

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

<header>
  <h1>Епідеміологічна модель</h1>
</header>

<nav>
  <a href="#model-description">Теоретична частина</a>
  <a href="#">Графік</a>
  <a href="#">Параметри</a>
  <a href="#">Експорт CSV</a>
  <a href="#">Експорт PDF</a>
</nav>

<div class="container">
  <!-- Ліва колонка (параметри) -->
  <div class="left-column">
    <h2>Налаштування моделі епідемії</h2>
    <form method="POST">

      {% for key, val in params.items() %}
        <label>{{ key }}:</label>
        <input name="{{ key }}" value="{{ val }}" type="number" step="any"><br>
      {% endfor %}
      <input type="submit" value="Запустити модель">
      
    </form>  

    <!-- порівняння результатів -->
    <br><br>
    <form method="POST" action="/compare">
      <input type="submit" value="Порівняти результати без нейромережі та з нейромережею">
    </form>

   
    </form>
  </div>

  <!-- Права колонка (графік) -->
  <div class="right-column">
    {% if plot_url %}
      <h3>Результат моделювання:</h3>
      <img src="data:image/png;base64,{{ plot_url }}" alt="Графік результатів моделювання">
      <section id="export" class="result">
        <p><a href="/download-csv">Завантажити CSV</a></p>
        <p><a href="/download-pdf">Завантажити PDF</a></p>
      </section>

      <h1>Помилки (для нейромережі)</h1>
                <p>Loss     : {{ mean_errors["Loss"] }}</p>
                <p>MAE      : {{ mean_errors["MAE"] }}</p>
                <p>MSE      : {{ mean_errors["MSE"] }}</p>
                <p>Root MSE : {{ mean_errors["RMSE"] }}</p>

    {% endif %}
  </div>
</div>

    

    <section id="model-description" class="description-container">

    <h3>Опис теоретичної моделі</h3>
    <p>Ми використовуємо стохастичну модель для моделювання поширення інфекційної хвороби в популяції, з урахуванням вакцинування та реінфекції. Ця модель складається з чотирьох груп осіб: сприйнятливі (S), інфіковані (I), одужавші (R), та вакциновані (V). Моделювання базується на стохастичних диференціальних рівняннях, де враховано ймовірнісні збурення для кожної групи.</p>
    
    <h4>Математичні рівняння:</h4>
    <p>Математичні рівняння моделі виглядають наступним чином:</p>
    <p>1. Для групи сприйнятливих:</p>
    <p>
      \( dS = - &#946; \cdot S \cdot I \cdot dt + \sigma \cdot S \cdot dW_S - v_{rate} \cdot S \cdot dt + reinf_{rate} \cdot R \cdot dt \)
    </p>
    <p>2. Для групи інфікованих:</p>
    <p>
      \( dI = ( &#946; \cdot S \cdot I - \gamma \cdot I) \cdot dt + \sigma \cdot I \cdot dW_I + {&#955;}_{jump} dN_t + {reinf}_{rate} \cdot R \cdot dt \cdot {reinf}_{factor} \)
    </p>
    <p>3. Для групи одужалих:</p>
    <p>
      \( dR = \gamma \cdot I \cdot dt + \sigma \cdot I \cdot dW_R - {reinf}_{rate} \cdot R \cdot dt \cdot {reinf}_{factor} \)
    </p>
    <p>4. Для групи вакцинованих:</p>
    <p>
      \( dV = v_{rate} \cdot S \cdot dt \) (починаючи з \( vaccine_{start}\) ) 
    </p>
    
    <h4>Опис змінних та параметрів:</h4>
    <ul>
      <li><strong>\( S \):</strong> Кількість осіб, які є сприйнятливими до інфекції.</li>
      <li><strong>\( I \):</strong> Кількість інфікованих осіб.</li>
      <li><strong>\( R \):</strong> Кількість осіб, які одужали від інфекції.</li>
      <li><strong>\( V \):</strong> Кількість вакцинованих осіб.</li>
      <li><strong>\( &#946; \):</strong> Коефіцієнт передачі інфекції (шанс передачі інфекції від інфікованого до сприйнятливого).</li>
      <li><strong>\(\gamma\):</strong> Коефіцієнт одужання (швидкість, з якою інфіковані одужують).</li>
      <li><strong>\(\sigma\):</strong> Параметр стохастичних збурень для кожної групи (генерація випадкових змінних).</li>
      <li><strong>\( &#955;_{jump} \):</strong> Середнє число випадкових "стрибків" (Пуассонівське збурення) для інфікованих.</li>
      <li><strong>\( v_{rate} \):</strong> Швидкість вакцинування.</li>
      <li><strong>\( reinf_{rate} \):</strong> Швидкість реінфекції (перехід від R до I).</li>
      <li><strong>\( reinf_{factor} \):</strong> Фактор, який визначає вплив реінфекції на популяцію.</li>
      <li><strong>\( t_{max} \):</strong> Максимальний час моделювання.</li>
      <li><strong>\( dt \):</strong> Крок часу для числового розв'язку.</li>
      <li><strong>\( vaccine_{start} \):</strong> Час початку вакцинування.</li>
    </ul>

    <h4>Інтерпретація моделі:</h4>
    <p>Ця модель моделює зміни в чисельності кожної групи через стохастичні диференціальні рівняння. Вакцинування поступово зменшує кількість сприйнятливих осіб і збільшує кількість вакцинованих, що дає змогу контролювати епідемію. Реінфекція та стохастичні збурення додають випадковість в процес і роблять модель більш реалістичною.</p>
  </section>

  <script type="text/javascript">
     MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
  </script>

'''

#<footer>
#  <p>Епідеміологічна модель 2025</p>
#</footer>


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

# Глобальна змінна для зберігання результату
plot_url = None

# Функція для обробки моделювання
def run_simulation_async(**params):
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
        model.fit(X, y, epochs=50, batch_size=32)

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
    plot_url = None
    params = DEFAULT_PARAMS.copy()
    print(params)
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
def long_running_task(params):
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
    plot_url = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()

    return plot_url

@app.route('/compare', methods=['GET', 'POST'])
def compare_results():
    global plot_url  # Використовуємо глобальну змінну

    # Запуск довготривалого обчислення в окремому потоці
    def update_plot_url():
        global plot_url  # Вказуємо на глобальну змінну
        plot_url = long_running_task(DEFAULT_PARAMS)

    # Стартуємо фоновий потік
    threading.Thread(target=update_plot_url, daemon=True).start()

    # Початковий шаблон (можна додати елементи, що показують статус, поки обчислення не завершені)
    return render_template_string("""
        <html>
        <head>
            <title>Порівняння результатів</title>
            <script>
                function loadPlot() {
                    fetch('/get_plot')
                        .then(response => response.json())
                        .then(data => {
                            if (data.plot_url) {
                                document.getElementById('plot_img').src = 'data:image/png;base64,' + data.plot_url;
                                document.getElementById('plot_img').style.display = 'block';
                            }
                        })
                        .catch(error => console.error('Error loading plot:', error));
                }
                setTimeout(loadPlot, 1000);  // Перевіряти кожну секунду
            </script>
        </head>
        <body>
            <h1>Порівняння результатів моделювання</h1>
            <p>Обчислення тривають...</p>
            <p>Будь ласка, зачекайте...</p>
            <img id="plot_img" alt="Графік" style="display:none;" />
        </body>
        </html>
    """)

@app.route('/get_plot')
def get_plot():
    # Повертаємо результат графіка, якщо він є
    if plot_url:
        return jsonify({'plot_url': plot_url})
    return jsonify({'plot_url': None})




if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
