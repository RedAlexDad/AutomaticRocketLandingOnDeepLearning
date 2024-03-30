import ctypes
import numpy as np

# Загрузка библиотеки
lib = ctypes.CDLL(
    '/home/redalexdad/Документы/GitHub/AutomaticRocketLandingOnDeepLearning/cpp/cmake-build-debug/libAutomaticRocketLandingOnDeepLearning.so'
)

# Определение типов аргументов и возвращаемого значения
lib.train_network.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_bool), ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)
]

# Класс для хранения данных сети
class NetworkData:
    def __init__(self, weights):
        self.weights = weights
        self.num_weights = len(weights)
        self._weights_ptr = None  # Указатель на память для весов

    def allocate_memory(self):
        # Выделение памяти для весов сети в C++
        self._weights_ptr = (ctypes.c_float * self.num_weights)(*self.weights)

    def release_memory(self):
        # Освобождение памяти для весов сети в C++
        # Дополните этот метод при необходимости
        pass


# Внутри метода train
def train(self):
    # Преобразование данных Python в типы C
    state_batch_c = (ctypes.c_float * len(state_batch))(*state_batch)
    action_batch_c = (ctypes.c_int * len(action_batch))(*action_batch)
    reward_batch_c = (ctypes.c_float * len(reward_batch))(*reward_batch)
    new_state_batch_c = (ctypes.c_float * len(new_state_batch))(*new_state_batch)
    done_batch_c = (ctypes.c_bool * len(done_batch))(*done_batch)

    # Создание экземпляра NetworkData
    network_data = NetworkData(self.q_net.get_weights())
    network_data.allocate_memory()

    # Вызов функции C
    updated_weights_ptr = lib.train_network(state_batch_c, action_batch_c, reward_batch_c, new_state_batch_c,
                                            done_batch_c, self.batch_size, ctypes.byref(network_data))

    # Проверка успешности вызова функции
    if updated_weights_ptr is not None:
        # Обновление весов сети Python из памяти C++
        updated_weights = np.ctypeslib.as_array(ctypes.cast(updated_weights_ptr, ctypes.POINTER(ctypes.c_float)),
                                                shape=(network_data.num_weights,))
        self.q_net.set_weights(updated_weights)

    # Освобождение памяти
    network_data.release_memory()

    # Дополнительный код

# Дополнительный код