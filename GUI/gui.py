import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image
import os
import random
import ttkbootstrap
import customtkinter as ctk
import io
import base64
import sys

sys.path.append(".")
import predictor as pred

# For the UI
ctk.set_default_color_theme("dark-blue")

flashcards_data = [
    {
        "character": "𓅐",
        "definition": "Vulture",
        "translation": "G14 - the glottal stop heard at the commencement of German words beginning with a vowel, ex. der Adler.",
    },
    {
        "character": "𓇋",
        "definition": "Reed leaf",
        "translation": "M17 - usually consonantal y; at the beginning of words sometimes identical with ꜣ.",
    },
    {
        "character": "𓏭",
        "definition": "Two diagonal strokes",
        "translation": "Z4 - y",
    },
    {
        "character": "𓂝",
        "definition": "Arm",
        "translation": "D21 - A guttural sound unknown to English.",
    },
    {
        "character": "𓅱",
        "definition": "Quail chick, var. Z7",
        "translation": "G43 - w",
    },
    {
        "character": "𓃀",
        "definition": "Foot",
        "translation": "D58 - b",
    },
    {
        "character": "𓊪",
        "definition": "Stool",
        "translation": "D58 - p",
    },
    {
        "character": "𓆑",
        "definition": "Horned viper",
        "translation": "I9 - f",
    },
    {
        "character": "𓅓",
        "definition": "Owl",
        "translation": "G17 - m",
    },
    {
        "character": "𓈖",
        "definition": "Water ripple",
        "translation": "N35 - n",
    },
    {
        "character": "𓂋",
        "definition": "Mouth",
        "translation": "D21 - r",
    },
    {
        "character": "𓉔",
        "definition": "Reed shelter",
        "translation": "O4 - h as in English",
    },
    {
        "character": "𓎛",
        "definition": "Wick",
        "translation": "V28 - emphatic h",
    },
    {
        "character": "𓐍",
        "definition": "Placenta",
        "translation": "Aa1 - like ch in Scotch loch",
    },
    {
        "character": "𓄡",
        "definition": "Animal belly and tail",
        "translation": "F32 - perhaps like ch in German ich",
    },
    {
        "character": "𓋴",
        "definition": "Folded cloth",
        "translation": "S29 - s",
    },
    {
        "character": "𓈙",
        "definition": "Pool",
        "translation": "N37 - sh",
    },
    {
        "character": "𓈎",
        "definition": "Sandy slope",
        "translation": "N29 - backward k; rather like our q in queen",
    },
    {
        "character": "𓎡",
        "definition": "Basket with handle",
        "translation": "V31 - k",
    },
    {
        "character": "𓎼",
        "definition": "Ring stand",
        "translation": "W11 - hard g",
    },
    {
        "character": "𓏏",
        "definition": "Small bread loaf",
        "translation": "X1 - t",
    },
    {
        "character": "𓍿",
        "definition": "Tethering rope",
        "translation": "V13 - originally ash (č or tj)",
    },
    {
        "character": "𓂧",
        "definition": "Hand",
        "translation": "D46 - d",
    },
    {
        "character": "𓆓",
        "definition": "Cobra",
        "translation": "I10 - originally dj and also a dull emphatic s",
    },
    {
        "character": "𓂃",
        "definition": "Eyebrow",
        "translation": "D13 - ḏayp",
    },
]


def round_rectangle(canvas, x1, y1, x2, y2, radius=20, **kwargs):
    """Draw a rounded rectangle on a Canvas and return its id."""
    points = [
        x1 + radius,
        y1,
        x2 - radius,
        y1,
        x2,
        y1,
        x2,
        y1 + radius,
        x2,
        y2 - radius,
        x2,
        y2,
        x2 - radius,
        y2,
        x1 + radius,
        y2,
        x1,
        y2,
        x1,
        y2 - radius,
        x1,
        y1 + radius,
        x1,
        y1,
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)


# Definition Page
class DefinitionPage(ttk.Frame):
    def __init__(self, master, controller):
        super().__init__(master)

        self.title_label = ttk.Label(
            self,
            text="Definition",
            font=("Helvetica", 24, "bold"),
        )
        self.title_label.pack(pady=50)

        self.result_label = ttk.Label(
            self,
            text="[Analysis Result and Definition will appear here]",
            font=("Arial", 16, "bold"),
        )
        self.result_label.pack(pady=40, padx=20)

        ctk.CTkButton(
            self,
            text="Back to Drawing Pad",
            command=lambda: controller.show_frame(HandwritingPad),
            border_width=2,
            border_color="#6fb9f6",
            fg_color="#6fb9f6",
            text_color="#ffffff",
            width=100,
        ).pack(pady=30, ipadx=10, ipady=5)

    def display_result(self, result_text):
        self.result_label.configure(text=result_text)


# Flashcards Page
class FlashcardsPage(ttk.Frame):
    def __init__(self, master, controller):
        super().__init__(master)
        self.controller = controller

        self.current_card_index = 0
        self.is_front_side = True

        ttk.Label(
            self,
            text="Flashcards Review Mode",
            font=("Helvetica", 24, "bold"),
        ).pack(pady=30)

        self.card_display_frame = tk.Frame(
            self,
            relief=tk.SOLID,
            borderwidth=1,
            padx=20,
            pady=30,
            bg="#FFFFFF",
            width=750,
            height=300,
        )
        self.card_display_frame.pack(pady=30, padx=80, fill="both", expand=True)
        self.card_display_frame.pack_propagate(False)

        self.card_content_label = tk.Label(
            self.card_display_frame,
            text="",
            font=("Arial", 40, "bold"),
            anchor="center",
            wraplength=400,
            justify=tk.CENTER,
            background="#FFFFFF",
            foreground="black",
        )
        self.card_content_label.pack(expand=True, fill="both", ipady=50)

        self.card_count_label = ttk.Label(
            self,
            text="",
            font=("Arial", 12),
        )
        self.card_count_label.pack(pady=5)

        control_frame_nav = ttk.Frame(self)
        control_frame_nav.pack(pady=10)

        ctk.CTkButton(
            control_frame_nav,
            text="<",
            command=self.show_prev_card,
            corner_radius=8,
            fg_color="transparent",
            border_width=2,
            border_color="#ffc107",
            text_color="#ffc107",
            hover_color="#1a1a1a",
            width=30,
        ).pack(side=tk.LEFT, padx=10)

        ctk.CTkButton(
            control_frame_nav,
            text="Flip Card",
            command=self.flip_card,
            border_width=2,
            border_color="#d453ff",
            fg_color="#d453ff",
            text_color="#ffffff",
            width=100,
        ).pack(side=tk.LEFT, padx=10)

        ctk.CTkButton(
            control_frame_nav,
            text=">",
            command=self.show_next_card,
            corner_radius=8,
            fg_color="transparent",
            border_width=2,
            border_color="#ffc107",
            text_color="#ffc107",
            hover_color="#1a1a1a",
            width=30,
        ).pack(side=tk.LEFT, padx=10)

        control_frame_action = ttk.Frame(self)
        control_frame_action.pack(pady=10)

        ctk.CTkButton(
            control_frame_action,
            text="🔀 Shuffle Cards",
            command=self.shuffle_cards,
            corner_radius=8,
            border_width=2,
            border_color="#00a2ff",
            text_color="#ffffff",
            width=30,
        ).pack(side=tk.LEFT, padx=10)

        ctk.CTkButton(
            control_frame_action,
            text="🏠 Home",
            command=lambda: controller.show_frame(HomePage),
            corner_radius=8,
            border_width=2,
            border_color="#00a2ff",
            text_color="#ffffff",
            width=30,
        ).pack(side=tk.LEFT, padx=10)

        self.update_card_display()

    def update_card_display(self):
        global flashcards_data
        if not flashcards_data:
            self.card_content_label.configure(
                text="No flashcards loaded.", background="#FFFFFF", foreground="red"
            )
            self.card_count_label.configure(text="")
            return

        self.card_display_frame.configure(bg="#FFFFFF")
        card = flashcards_data[self.current_card_index]

        if self.is_front_side:
            text = card["character"]
            fg_color = "black"
            font = ("Arial", 100, "bold")
            wrap = 800
            pady = 40
        else:
            text = card["definition"] + "\n" + card["translation"]
            fg_color = "#006400"
            font = ("Arial", 40, "bold")
            wrap = 600
            pady = 10

        self.card_content_label.configure(
            text=text,
            background="#FFFFFF",
            foreground=fg_color,
            font=font,
            wraplength=wrap,
            pady=pady,
        )

        self.card_content_label.configure(
            text=text,
            background="#FFFFFF",
            foreground=fg_color,
        )
        count_text = f"Card {self.current_card_index + 1} of {len(flashcards_data)}"
        self.card_count_label.configure(text=count_text)

    def flip_card(self):
        self.is_front_side = not self.is_front_side
        self.update_card_display()

    def show_next_card(self):
        global flashcards_data
        if not flashcards_data:
            return
        self.current_card_index = (self.current_card_index + 1) % len(flashcards_data)
        self.is_front_side = True
        self.update_card_display()

    def show_prev_card(self):
        global flashcards_data
        if not flashcards_data:
            return
        self.current_card_index = (
            self.current_card_index - 1 + len(flashcards_data)
        ) % len(flashcards_data)
        self.is_front_side = True
        self.update_card_display()

    def shuffle_cards(self):
        global flashcards_data
        if not flashcards_data:
            messagebox.showwarning("Shuffle Failed", "No cards to shuffle.")
            return
        random.shuffle(flashcards_data)
        self.current_card_index = 0
        self.is_front_side = True
        self.update_card_display()
        messagebox.showinfo("Shuffle Complete", "Flashcards have been shuffled!")


# Drawing Pad
class HandwritingPad(ttk.Frame):
    def __init__(self, master, controller):
        super().__init__(master)
        self.controller = controller

        self.last_x, self.last_y = None, None
        self.drawing_color = "black"
        self.line_width = 3

        self.undo_history = []
        self.redo_history = []
        self.current_stroke_ids = []

        ttk.Label(self, text="📝 Drawing Pad", font=("Arial", 18, "bold")).pack(pady=10)

        self.canvas_wrapper = tk.Canvas(
            self,
            bg="#FFFFFF",
            highlightthickness=0,
            bd=0,
            width=540,
            height=540,
        )
        self.canvas_wrapper.pack(pady=10, padx=20)
        self.canvas_wrapper.pack_propagate(False)

        round_rectangle(
            self.canvas_wrapper,
            10,
            10,
            530,
            530,
            radius=40,
            fill="#FFFFFF",
            outline="",
        )

        self.canvas = tk.Canvas(
            self.canvas_wrapper,
            bg="white",
            width=500,
            height=500,
            highlightthickness=2,
            highlightbackground="#FFFFFF",
            bd=0,
        )
        self.canvas.configure(bg="white", insertbackground="white")

        self.canvas_wrapper.create_window(270, 270, window=self.canvas)

        self.canvas.bind("<Button-1>", self.start_stroke)
        self.canvas.bind("<B1-Motion>", self.continue_stroke)
        self.canvas.bind("<ButtonRelease-1>", self.end_stroke)

        control_frame = ttk.Frame(self)
        control_frame.pack(pady=15)

        btn_padding = (10, 6)

        self.home_button = ctk.CTkButton(
            control_frame,
            text="🏠 Home",
            command=lambda: controller.show_frame(HomePage),
            corner_radius=8,
            border_width=2,
            border_color="#00a2ff",
            text_color="#ffffff",
            hover_color="#1a1a1a",
            width=90,
        )
        self.home_button.pack(side=tk.LEFT, padx=5)

        self.undo_button = ctk.CTkButton(
            control_frame,
            text="↺ Undo",
            command=self.undo_stroke,
            corner_radius=8,
            fg_color="transparent",
            border_width=2,
            border_color="#ffc107",
            text_color="#ffc107",
            hover_color="#1a1a1a",
            width=90,
        )
        self.undo_button.pack(side=tk.LEFT, padx=5)

        self.redo_button = ctk.CTkButton(
            control_frame,
            text="↺ Redo",
            command=self.redo_stroke,
            corner_radius=8,
            fg_color="transparent",
            border_width=2,
            border_color="#ffc107",
            text_color="#ffc107",
            hover_color="#1a1a1a",
            width=90,
        )
        self.redo_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = ctk.CTkButton(
            control_frame,
            text="Clear Pad",
            command=self.clear_canvas,
            corner_radius=8,
            border_width=2,
            fg_color="#fc3030",
            border_color="#fc3030",
            text_color="#FFFFFF",
            width=90,
        )
        self.clear_button.pack(side=tk.LEFT, padx=10)

        self.analyze_button = ctk.CTkButton(
            control_frame,
            text="Analyze & Save",
            command=self.analyze_and_save,
            corner_radius=8,
            border_width=2,
            fg_color="#1ab834",
            border_color="#1ab834",
            text_color="#FFFFFF",
            width=90,
        )
        self.analyze_button.pack(side=tk.LEFT, padx=10)

    def _update_button_states(self):
        self.undo_button.configure(
            state=tk.NORMAL if self.undo_history else tk.DISABLED
        )
        self.redo_button.configure(
            state=tk.NORMAL if self.redo_history else tk.DISABLED
        )

    def start_stroke(self, event):
        if event.x < 0 or event.x > 500 or event.y < 0 or event.y > 500:
            messagebox.showerror("Error", "Please draw inside the canvas area.")
            return

        self.last_x = event.x
        self.last_y = event.y

    def continue_stroke(self, event):
        if self.last_x is not None and self.last_y is not None:
            line_id = self.canvas.create_line(
                self.last_x,
                self.last_y,
                event.x,
                event.y,
                fill=self.drawing_color,
                width=self.line_width,
                capstyle=tk.ROUND,
                joinstyle=tk.ROUND,
            )
            self.current_stroke_ids.append(line_id)
            self.last_x = event.x
            self.last_y = event.y

    def end_stroke(self, event):
        if self.current_stroke_ids:
            self.undo_history.append(self.current_stroke_ids)
            self.current_stroke_ids = []
            self.redo_history = []
            self._update_button_states()

        self.last_x, self.last_y = None, None

    def undo_stroke(self):
        if self.undo_history:
            stroke_to_undo = self.undo_history.pop()
            for item_id in stroke_to_undo:
                self.canvas.itemconfigure(item_id, state="hidden")

            self.redo_history.append(stroke_to_undo)
            self._update_button_states()

    def redo_stroke(self):
        if self.redo_history:
            stroke_to_redo = self.redo_history.pop()
            for item_id in stroke_to_redo:
                self.canvas.itemconfigure(item_id, state="normal")

            self.undo_history.append(stroke_to_redo)
            self._update_button_states()

    def clear_canvas(self):
        self.canvas.delete("all")
        self.undo_history = []
        self.redo_history = []
        self.current_stroke_ids = []
        self._update_button_states()

    def analyze_and_save(self):
        if not self.canvas.find_all():
            messagebox.showwarning(
                "Empty Canvas", "Please draw something before analyzing."
            )
            return

        temp_ps_file = "temp_drawing.ps"
        output_file = "handwriting_output.png"
        analysis_result = ""

        try:
            self.canvas.postscript(file=temp_ps_file, colormode="color")

            img = Image.open(temp_ps_file)
            if img.mode != "L":
                img = img.convert("L")
            png_buffer = io.BytesIO()
            img.save(png_buffer, format="PNG")
            output_file_name = "handwriting_output.txt"
            png_bytes = png_buffer.getvalue()
            base64_string = base64.b64encode(png_bytes).decode("utf-8")
            with open(output_file_name, "w") as f:
                f.write(base64_string)
            with open("handwriting_output.txt", "r") as f:
                base64_string = f.read()
            png_bytes = base64.b64decode(base64_string)
            img = Image.open(io.BytesIO(png_bytes))
            img.save("reconstructed.png")
            idx, info, probs = pred.predict_hieroglyph("reconstructed.png")
            analysis_result = (
                "Symbol: "
                + info["hieroglyph"]
                + "\t"
                + "Description: "
                + info["description"]
            )
            self.controller.frames[DefinitionPage.__name__].display_result(
                analysis_result
            )

            self.controller.show_frame(DefinitionPage)

            self.clear_canvas()

        except ImportError:
            messagebox.showerror(
                "Error: Pillow Not Found",
                "The 'Pillow' library is required to save the image. Please install it using: pip install Pillow",
            )
        except Exception as e:
            messagebox.showerror("Error Saving Image", f"An error occurred: {e}")
        finally:
            if os.path.exists(temp_ps_file):
                os.remove(temp_ps_file)


# Home Page
class HomePage(ttk.Frame):
    def __init__(self, master, controller):
        super().__init__(master)
        self.controller = controller

        title_label = ttk.Label(
            self,
            text="𓀀 ArchaeoHack Translator App",
            font=("Helvetica", 50, "bold"),
        )
        title_label.pack(pady=70, padx=70)

        desc_label = ttk.Label(
            self, text="Welcome! Choose an option below! (Only Unilateral Symbols)"
        )
        desc_label.pack(pady=10)

        button_frame = ttk.Frame(self)
        button_frame.pack(pady=20)

        start_button = ctk.CTkButton(
            button_frame,
            text="Start Drawing Pad \n\n\u270f\ufe0f",
            command=lambda: controller.show_frame(HandwritingPad),
            corner_radius=8,
            border_width=2,
            fg_color="#00a2ff",
            border_color="#00a2ff",
            text_color="#FFFFFF",
            width=350,
            font=("Helvetica", 24, "bold"),
        )
        start_button.pack(side=tk.LEFT, padx=10, ipadx=10, ipady=140)

        flashcards_button = ctk.CTkButton(
            button_frame,
            text="Flashcards \n\n\U0001f4da",
            command=lambda: controller.show_frame(FlashcardsPage),
            corner_radius=8,
            border_width=2,
            fg_color="#00a2ff",
            border_color="#00a2ff",
            text_color="#FFFFFF",
            width=350,
            font=("Helvetica", 24, "bold"),
        )
        flashcards_button.pack(side=tk.LEFT, padx=10, ipadx=10, ipady=140)


class PageManager(ttkbootstrap.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, themename="darkly")
        self.title("ArchaeoHack Translator")
        self.geometry("1100x700")

        self.configure(bg="#020617")

        style = ttk.Style()
        style.configure("HomePage.TFrame", background="#020617")
        style.configure(
            "HomeHero.TLabel",
            background="#020617",
            foreground="white",
            font=("Helvetica", 32, "bold"),
        )
        style.configure(
            "HomeSub.TLabel",
            background="#020617",
            foreground="#9ca3af",
            font=("Helvetica", 14),
        )
        container = ttk.Frame(self, style="HomePage.TFrame")
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (HomePage, HandwritingPad, FlashcardsPage, DefinitionPage):
            page_name = F.__name__
            frame = F(master=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(HomePage)

    def show_frame(self, page_class):
        frame = self.frames[page_class.__name__]
        frame.tkraise()


# Main Execution
if __name__ == "__main__":
    app = PageManager()
    app.mainloop()
