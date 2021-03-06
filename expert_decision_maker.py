import time
from pynput import keyboard


class KeyboardHandler:
    def __init__(self):
        self.__last_key = None
        self.__keyboard_listener = keyboard.Listener(on_press=self.__on_press)
        self.__keyboard_listener.start()

    def __del__(self):
        self.finalize()

    def finalize(self):
        self.__keyboard_listener.stop()

    def get_key(self):
        self.__last_key = None
        while self.__keyboard_listener.is_alive() and self.__last_key is None:
            time.sleep(0.01)
        return self.__last_key

    def __on_press(self, key):
        if key == keyboard.Key.esc:
            return False
        self.__last_key = key
        return True


class ExpertDecisionMaker:
    def __init__(self, env):
        self.__key_to_action = dict()
        self.__keyboard_handler = KeyboardHandler()
        self.__setup_keys(env.unwrapped.get_action_meanings())

    def __del__(self):
        self.finalize()

    def decide(self):
        key = None
        while key not in self.__key_to_action:
            key = self.__keyboard_handler.get_key()
            if key is None:
                return None
        return self.__key_to_action[key]

    def finalize(self):
        self.__keyboard_handler.finalize()

    def __setup_keys(self, actions):
        for action_index, action in enumerate(actions):
            print("{}) {} := ".format(action_index, action), end="", flush=True)
            key = self.__keyboard_handler.get_key()
            print(key)
            if key is None:
                return
            self.__key_to_action[key] = action_index
