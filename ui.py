import json
import os
import re
import threading
import tkinter as tk
from tkinter import filedialog, messagebox


DEFAULT_SUMMARY_TOKENS = 120
MIN_SUMMARY_TOKENS = 20
MAX_SUMMARY_TOKENS = 1000


class AINotetakerUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Notetaker - Text Summarizer")
        self.root.geometry("1000x850")
        self.root.minsize(720, 560)
        self.root.resizable(True, True)

        self.generating = False
        self.root.configure(bg="#f0f0f0")

        main_frame = tk.Frame(root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        input_frame = tk.LabelFrame(
            main_frame,
            text="Input Text",
            font=("Arial", 12, "bold"),
            bg="#f0f0f0",
            padx=10,
            pady=10,
        )
        input_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        input_scrollbar = tk.Scrollbar(input_frame)
        input_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.input_text = tk.Text(
            input_frame,
            height=10,
            font=("Arial", 10),
            yscrollcommand=input_scrollbar.set,
            wrap=tk.WORD,
            bg="white",
            fg="black",
            relief=tk.SUNKEN,
            borderwidth=1,
        )
        self.input_text.pack(fill=tk.BOTH, expand=True)
        input_scrollbar.config(command=self.input_text.yview)

        input_button_frame = tk.Frame(input_frame, bg="#f0f0f0")
        input_button_frame.pack(fill=tk.X, pady=(10, 0))

        load_button = tk.Button(
            input_button_frame,
            text="Load from File",
            command=self.load_file,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=10,
            pady=5,
            relief=tk.RAISED,
            cursor="hand2",
        )
        load_button.pack(side=tk.LEFT, padx=(0, 5))

        controls_frame = tk.Frame(main_frame, bg="#f0f0f0")
        controls_frame.pack(fill=tk.X, pady=(0, 10))

        length_frame = tk.Frame(controls_frame, bg="#f0f0f0")
        length_frame.pack(side=tk.LEFT, padx=(0, 20))

        tk.Label(
            length_frame,
            text="Summary Length:",
            font=("Arial", 10),
            bg="#f0f0f0",
        ).pack(side=tk.LEFT, padx=(0, 5))

        self.length_var = tk.IntVar(value=DEFAULT_SUMMARY_TOKENS)
        length_slider = tk.Scale(
            length_frame,
            from_=MIN_SUMMARY_TOKENS,
            to=MAX_SUMMARY_TOKENS,
            orient=tk.HORIZONTAL,
            variable=self.length_var,
            bg="#f0f0f0",
            fg="black",
            length=180,
            cursor="hand2",
        )
        length_slider.pack(side=tk.LEFT)

        self.length_label = tk.Label(
            length_frame,
            text=str(DEFAULT_SUMMARY_TOKENS),
            font=("Arial", 10, "bold"),
            bg="#f0f0f0",
            width=4,
        )
        self.length_label.pack(side=tk.LEFT, padx=(5, 0))
        length_slider.config(command=self.update_length_label)

        self.generate_button = tk.Button(
            controls_frame,
            text="Generate Summary",
            command=self.generate_summary_handler,
            bg="#2196F3",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=15,
            pady=8,
            relief=tk.RAISED,
            cursor="hand2",
        )
        self.generate_button.pack(side=tk.RIGHT, padx=(10, 0))

        self.status_label = tk.Label(
            controls_frame,
            text="Ready",
            font=("Arial", 9),
            bg="#f0f0f0",
            fg="#666666",
        )
        self.status_label.pack(side=tk.LEFT, padx=(20, 0))

        output_frame = tk.LabelFrame(
            main_frame,
            text="Generated Summary",
            font=("Arial", 12, "bold"),
            bg="#f0f0f0",
            padx=10,
            pady=10,
        )
        output_frame.pack(fill=tk.BOTH, expand=True)

        output_scrollbar = tk.Scrollbar(output_frame)
        output_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.output_text = tk.Text(
            output_frame,
            height=15,
            font=("Arial", 10),
            yscrollcommand=output_scrollbar.set,
            wrap=tk.WORD,
            bg="white",
            fg="black",
            relief=tk.SUNKEN,
            borderwidth=1,
            state=tk.NORMAL,
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)
        output_scrollbar.config(command=self.output_text.yview)

        output_button_frame = tk.Frame(output_frame, bg="#f0f0f0")
        output_button_frame.pack(fill=tk.X, pady=(10, 0))

        copy_button = tk.Button(
            output_button_frame,
            text="Copy to Clipboard",
            command=self.copy_to_clipboard,
            bg="#FF9800",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=10,
            pady=5,
            relief=tk.RAISED,
            cursor="hand2",
        )
        copy_button.pack(side=tk.LEFT, padx=(0, 5))

        save_button = tk.Button(
            output_button_frame,
            text="Save to File",
            command=self.save_file,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=10,
            pady=5,
            relief=tk.RAISED,
            cursor="hand2",
        )
        save_button.pack(side=tk.LEFT, padx=(0, 5))

        clear_button = tk.Button(
            output_button_frame,
            text="Clear All",
            command=self.clear_all,
            bg="#f44336",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=10,
            pady=5,
            relief=tk.RAISED,
            cursor="hand2",
        )
        clear_button.pack(side=tk.LEFT)

    def update_length_label(self, value):
        self.length_label.config(text=str(int(float(value))))

    def generate_summary_handler(self):
        if self.generating:
            messagebox.showwarning("In Progress", "Summary generation is already in progress. Please wait.")
            return

        input_text = self.input_text.get("1.0", tk.END).strip()
        if not input_text:
            messagebox.showwarning("Empty Input", "Please enter some text to summarize.")
            return

        self.generating = True
        self.generate_button.config(state=tk.DISABLED)
        self.status_label.config(text="Loading model and generating...", fg="#FF9800")
        self.root.update_idletasks()

        thread = threading.Thread(target=self._generate_in_background, args=(input_text,), daemon=True)
        thread.start()

    def _generate_in_background(self, text):
        try:
            from main import generate_summary

            max_length = self.length_var.get()
            summary = generate_summary(text, max_length=max_length)
            self.root.after(0, self._update_output, summary)
        except Exception as e:
            self.root.after(0, self._show_error, str(e))

    def _update_output(self, summary):
        cleaned_summary = self._clean_summary_text(summary)

        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", cleaned_summary)

        self.status_label.config(text="Summary generated successfully", fg="#4CAF50")
        self.generate_button.config(state=tk.NORMAL)
        self.generating = False

    def _clean_summary_text(self, text):
        text = self._repair_text_encoding(str(text).strip())

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                for key in ["text", "summary", "content", "generated_text"]:
                    if key in parsed:
                        return self._repair_text_encoding(str(parsed[key]).strip())
                for value in parsed.values():
                    if isinstance(value, str) and value.strip():
                        return self._repair_text_encoding(value.strip())
            elif isinstance(parsed, list) and parsed:
                first_item = parsed[0]
                if isinstance(first_item, dict):
                    for key in ["text", "summary", "content", "generated_text"]:
                        if key in first_item:
                            return self._repair_text_encoding(str(first_item[key]).strip())
                return self._repair_text_encoding(str(first_item).strip())
        except (json.JSONDecodeError, ValueError):
            pass

        match = re.search(r"\[\s*['\"]?text['\"]?\s*:\s*['\"]?([^'\"]*?)['\"]?\s*\]", text)
        if match:
            return self._repair_text_encoding(match.group(1).strip())

        match = re.search(r"\{\s*['\"]?text['\"]?\s*:\s*['\"]?([^'\"]*?)['\"]?\s*\}", text)
        if match:
            return self._repair_text_encoding(match.group(1).strip())

        if text.startswith("[") or text.startswith("{"):
            cleaned = text.strip("[]{}").strip("'\"")
            return self._repair_text_encoding(cleaned if cleaned else text)

        return text

    def _repair_text_encoding(self, text):
        mojibake_markers = (
            "\u00c3",
            "\u00c2",
            "\u00e2\u20ac",
            "\u00e2\u20ac\u2122",
            "\u00e2\u20ac\u0153",
            "\u00f0\u0178",
        )
        if not any(marker in text for marker in mojibake_markers):
            return text

        best_text = text
        best_score = sum(best_text.count(marker) for marker in mojibake_markers)

        for encoding in ("cp1252", "latin-1"):
            try:
                candidate = text.encode(encoding, errors="ignore").decode("utf-8")
            except UnicodeError:
                continue

            score = sum(candidate.count(marker) for marker in mojibake_markers)
            if score < best_score:
                best_text = candidate
                best_score = score

        return best_text

    def _read_text_file(self, file_path):
        for encoding in ("utf-8-sig", "utf-8", "cp1252"):
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return self._repair_text_encoding(f.read())
            except UnicodeDecodeError:
                continue

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return self._repair_text_encoding(f.read())

    def _show_error(self, error_msg):
        self.status_label.config(text=f"Error: {error_msg[:50]}", fg="#f44336")
        self.generate_button.config(state=tk.NORMAL)
        self.generating = False
        messagebox.showerror("Error", f"Failed to generate summary:\n\n{error_msg}")

    def load_file(self):
        file_path = filedialog.askopenfilename(
            title="Load Text File",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
            initialdir=os.getcwd(),
        )

        if not file_path:
            return

        try:
            content = self._read_text_file(file_path)
            self.input_text.delete("1.0", tk.END)
            self.input_text.insert("1.0", content)
            self.status_label.config(text=f"Loaded: {os.path.basename(file_path)}", fg="#4CAF50")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
            self.status_label.config(text="Failed to load file", fg="#f44336")

    def save_file(self):
        summary_text = self.output_text.get("1.0", tk.END).strip()
        if not summary_text:
            messagebox.showwarning("Empty Output", "No summary to save. Generate a summary first.")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Summary",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
            defaultextension=".txt",
            initialfile="summary.txt",
            initialdir=os.getcwd(),
        )

        if not file_path:
            return

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(summary_text)

            messagebox.showinfo("Success", f"Summary saved to:\n{file_path}")
            self.status_label.config(text=f"Saved: {os.path.basename(file_path)}", fg="#4CAF50")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file:\n{str(e)}")
            self.status_label.config(text="Failed to save file", fg="#f44336")

    def copy_to_clipboard(self):
        summary_text = self.output_text.get("1.0", tk.END).strip()
        if not summary_text:
            messagebox.showwarning("Empty Output", "No summary to copy.")
            return

        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(summary_text)
            self.root.update()
            messagebox.showinfo("Success", "Summary copied to clipboard!")
            self.status_label.config(text="Copied to clipboard", fg="#4CAF50")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy to clipboard:\n{str(e)}")
            self.status_label.config(text="Failed to copy", fg="#f44336")

    def clear_all(self):
        confirm = messagebox.askyesno("Confirm Clear", "Are you sure you want to clear all text?")
        if confirm:
            self.input_text.delete("1.0", tk.END)
            self.output_text.delete("1.0", tk.END)
            self.status_label.config(text="Cleared", fg="#666666")


def main():
    root = tk.Tk()
    AINotetakerUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
