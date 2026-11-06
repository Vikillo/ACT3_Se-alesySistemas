
# ACTIVIDAD: Implementación y evaluación de filtros digitales

# Importamos las librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from scipy.signal import sosfreqz
from scipy.signal import butter, sosfiltfilt, sosfreqz, firwin, filtfilt, freqz

# 1. DEFINICIÓN DE LA SEÑAL DE ENTRADA

# Frecuencia de muestreo y duración de la señal
fs = 500
duracion = 2

# Creamos el tiempo de la señal
t = np.arange(0, duracion, 1/fs)

# Creamos tres señales con diferentes frecuencias
senal1 = np.sin(2 * np.pi * 5 * t)
senal2 = np.sin(2 * np.pi * 30 * t)
senal3 = np.sin(2 * np.pi * 80 * t)

# Unimos las tres señales
senal_original = senal1 + senal2 + senal3

# Agregamos ruido blanco
np.random.seed(42)
ruido = np.random.normal(0, 0.5, len(t))

# Señal final con ruido
senal_ruidosa = senal_original + ruido


# 2. DISEÑO DE LOS FILTROS

# Filtro pasa bajos: permite frecuencias menores a 15 Hz
pasa_bajos = butter(
    4, 15, btype='low', fs=fs, output='sos'
)

# Filtro pasa altos: permite frecuencias mayores a 40 Hz
pasa_altos = butter(
    4, 40, btype='high', fs=fs, output='sos'
)

# Filtro pasa bandas: permite frecuencias entre 20 y 40 Hz
pasa_bandas = butter(
    4, [20, 40], btype='band', fs=fs, output='sos'
)


# 3. APLICACIÓN DE LOS FILTROS

resultado_bajos = sosfiltfilt(pasa_bajos, senal_ruidosa)
resultado_altos = sosfiltfilt(pasa_altos, senal_ruidosa)
resultado_bandas = sosfiltfilt(pasa_bandas, senal_ruidosa)


# 4. VISUALIZACIÓN DE LOS RESULTADOS

plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(t, senal_ruidosa)
plt.title("Señal original con ruido")
plt.ylabel("Amplitud")
plt.grid()

plt.subplot(4, 1, 2)
plt.plot(t, resultado_bajos)
plt.title("Resultado del filtro pasa bajos")
plt.ylabel("Amplitud")
plt.grid()

plt.subplot(4, 1, 3)
plt.plot(t, resultado_altos)
plt.title("Resultado del filtro pasa altos")
plt.ylabel("Amplitud")
plt.grid()

plt.subplot(4, 1, 4)
plt.plot(t, resultado_bandas)
plt.title("Resultado del filtro pasa bandas")
plt.xlabel("Tiempo (segundos)")
plt.ylabel("Amplitud")
plt.grid()

plt.tight_layout()
plt.show()


# 5. ANALISIS DE LA RESPUESTA EN FRECUENCIA

# Guardamos los filtros para analizarlos
filtros = [
    ("Filtro pasa bajos", pasa_bajos),
    ("Filtro pasa altos", pasa_altos),
    ("Filtro pasa bandas", pasa_bandas)
]

plt.figure(figsize=(12, 9))

# Analizamos cada filtro
for i, (nombre, filtro) in enumerate(filtros):

    frecuencia, respuesta = sosfreqz(
        filtro, worN=2048, fs=fs
    )

    # Convertimos la respuesta a decibelios
    magnitud = 20 * np.log10(
        np.maximum(np.abs(respuesta), 1e-10)
    )

    plt.subplot(3, 1, i + 1)
    plt.plot(frecuencia, magnitud)

    plt.title(nombre)
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud (dB)")
    plt.grid()

plt.tight_layout()
plt.show()


# 6. DISEÑO DEL FILTRO FIR

# Número de coeficientes del filtro
num_coeficientes = 101

# Diseñamos un filtro pasa bajos utilizando ventana Hamming
filtro_fir = firwin(
    num_coeficientes,
    cutoff=15,
    fs=fs,
    window='hamming'
)

# Aplicamos el filtro FIR a la señal con ruido
resultado_fir = filtfilt(
    filtro_fir,
    1.0,
    senal_ruidosa
)


# 7. COMPARACIÓN ENTRE FILTROS FIR E IIR

plt.figure(figsize=(12, 7))

plt.subplot(2, 1, 1)
plt.plot(t, resultado_bajos, label="Filtro IIR")
plt.plot(t, resultado_fir, label="Filtro FIR", alpha=0.8)

plt.title("Comparación de filtros pasa bajos FIR e IIR")
plt.ylabel("Amplitud")
plt.legend()
plt.grid()

# Comparamos sus respuestas en frecuencia
frecuencia_iir, respuesta_iir = sosfreqz(
    pasa_bajos, worN=2048, fs=fs
)

frecuencia_fir, respuesta_fir = freqz(
    filtro_fir, worN=2048, fs=fs
)

# Convertimos las magnitudes a decibelios
magnitud_iir = 20 * np.log10(
    np.maximum(np.abs(respuesta_iir), 1e-10)
)

magnitud_fir = 20 * np.log10(
    np.maximum(np.abs(respuesta_fir), 1e-10)
)

plt.subplot(2, 1, 2)
plt.plot(frecuencia_iir, magnitud_iir, label="Filtro IIR")
plt.plot(frecuencia_fir, magnitud_fir, label="Filtro FIR")

plt.title("Respuesta en frecuencia: FIR e IIR")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud (dB)")
plt.xlim(0, 100)
plt.ylim(-100, 5)
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
