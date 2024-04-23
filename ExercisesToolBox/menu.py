import os
import tkinter as tk
from tkinter import messagebox

def execute_script(exercise):
    if exercise == "Lagartijas":
        exercise_script = "push_ups.py"
    elif exercise == "Sentadillas":
        exercise_script = "squatsOriginal.py"
    # else:
        # exercise_script = exercise #.lower().replace(" ", "_") + ".py"
    # exercise_path = os.path.join("ExercisesToolBox", exercise_script)  # Usar ruta relativa

    if os.path.exists(exercise_script):
        os.system(f"python {exercise_script}")
    else:
        messagebox.showerror("Error", "El script del ejercicio seleccionado no existe.")

def main():
    options = [
        "Lagartijas",
        "Sentadillas",
        "ABS",
    ]

    root = tk.Tk()
    root.title("Selector de Ejercicios")
    root.geometry("800x700")

    label = tk.Label(root, text="Seleccione el ejercicio que desea detectar:", font=("Arial", 18), fg="black")
    label.pack(pady=10)

    listbox = tk.Listbox(root, font=("Arial", 18), fg="grey", selectbackground="lightblue")
    for option in options:
        listbox.insert(tk.END, option)
    listbox.pack(pady=5)

    def on_select():
        selected_index = listbox.curselection()
        if selected_index:
            selected_option = options[selected_index[0]]
            execute_script(selected_option)

    button = tk.Button(root, text="Ejecutar", font=("Arial", 15), bg="green", fg="white", command=on_select)
    button.pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    main()
