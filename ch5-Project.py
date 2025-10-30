import os
import cv2
import numpy as np
from tkinter import Tk, Button, Label, Frame, StringVar, messagebox, simpledialog
from PIL import Image, ImageTk
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

# ---------------- CONFIG ----------------
DATASET_PATH = "faces_dataset"
ADMIN_PASSWORD = "admin123"
IMG_SIZE = (100, 100)
# ----------------------------------------

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition (Stable Version)")
        self.root.geometry("800x600")

        self.cap = None
        self.recognizing = False
        self.pca = None
        self.knn = None
        self.names = []
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        os.makedirs(DATASET_PATH, exist_ok=True)
        self.setup_ui()

    def setup_ui(self):
        self.frame_buttons = Frame(self.root)
        self.frame_buttons.pack(pady=10)

        Button(self.frame_buttons, text="➕ Add New Face", width=20, command=self.add_new_face).grid(row=0, column=0, padx=8)
        Button(self.frame_buttons, text="🔁 Train Model", width=20, command=self.train_model_ui).grid(row=0, column=1, padx=8)
        self.recognize_btn = Button(self.frame_buttons, text="🎥 Start Recognition", width=20, command=self.toggle_recognition)
        self.recognize_btn.grid(row=0, column=2, padx=8)
        Button(self.frame_buttons, text="🚪 Exit", width=15, command=self.exit_app).grid(row=0, column=3, padx=8)

        self.video_frame = Frame(self.root, bd=2, relief="sunken")
        self.video_frame.pack()
        self.video_label = Label(self.video_frame)
        self.video_label.pack()

        self.status_var = StringVar()
        Label(self.root, textvariable=self.status_var, anchor="w").pack(fill="x", padx=10, pady=10)
        self.set_status("Idle")

    def set_status(self, text):
        self.status_var.set("Status: " + text)
        self.root.update_idletasks()

    # ---------- Camera ----------
    def open_camera(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Cannot access camera.")
                return False
        return True

    def close_camera(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None

    # ---------- Dataset ----------
    def add_new_face(self):
        pwd = simpledialog.askstring("Admin Login", "Enter admin password:", show='*')
        if pwd != ADMIN_PASSWORD:
            messagebox.showerror("Access Denied", "Wrong password.")
            return
        name = simpledialog.askstring("Add Person", "Enter person's name:")
        if not name:
            return

        path = os.path.join(DATASET_PATH, name)
        os.makedirs(path, exist_ok=True)
        if not self.open_camera():
            return

        captured = 0
        while captured < 3:
            ret, frame = self.cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                roi = gray[y:y+h, x:x+w]
                roi = cv2.resize(roi, IMG_SIZE)
                cv2.imwrite(os.path.join(path, f"{captured+1}.jpg"), roi)
                captured += 1
                self.set_status(f"Captured {captured}/3 images for {name}")
            self.show_frame(frame)
            self.root.update()

        messagebox.showinfo("Done", f"Saved 3 images for {name}.")
        self.set_status("Capture complete.")
        self.close_camera()

    def load_dataset(self):
        images, labels, names = [], [], []
        people = [p for p in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, p))]
        for label, person in enumerate(people):
            person_path = os.path.join(DATASET_PATH, person)
            for file in os.listdir(person_path):
                img_path = os.path.join(person_path, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None or img.size == 0:
                    continue
                img = cv2.resize(img, IMG_SIZE)
                img = img.astype(np.float32) / 255.0
                images.append(img.flatten())
                labels.append(label)
            names.append(person)
        if len(images) == 0:
            return None, None, None
        return np.array(images), np.array(labels), names

    def train_model_ui(self):
        self.set_status("Training model...")
        data = self.load_dataset()
        if data[0] is None:
            messagebox.showwarning("No Data", "No valid images found. Add faces first.")
            self.set_status("Idle")
            return

        images, labels, names = data
        n_components = min(20, len(images), images.shape[1])
        try:
            pca = PCA(n_components=n_components, whiten=True, svd_solver='randomized')
            X_pca = pca.fit_transform(images)
            X_pca = np.nan_to_num(X_pca)  # remove NaNs
            knn = KNeighborsClassifier(n_neighbors=min(3, len(np.unique(labels))))
            knn.fit(X_pca, labels)
            self.pca, self.knn, self.names = pca, knn, names
            messagebox.showinfo("Success", f"Trained on {len(names)} person(s).")
            self.set_status("Model trained successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Training failed:\n{str(e)}")
            self.set_status("Training failed.")

    # ---------- Recognition ----------
    def toggle_recognition(self):
        if not self.recognizing:
            if self.pca is None or self.knn is None:
                self.train_model_ui()
                if self.pca is None:
                    return
            if not self.open_camera():
                return
            self.recognizing = True
            self.recognize_btn.config(text="⛔ Stop Recognition")
            self.set_status("Recognition running...")
            self.recognize_loop()
        else:
            self.recognizing = False
            self.recognize_btn.config(text="🎥 Start Recognition")
            self.set_status("Recognition stopped.")
            self.close_camera()

    def recognize_loop(self):
        if not self.recognizing:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(15, self.recognize_loop)
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, IMG_SIZE).astype(np.float32) / 255.0
            roi_flat = roi.flatten().reshape(1, -1)
            face_pca = self.pca.transform(roi_flat)
            pred = self.knn.predict(face_pca)
            name = self.names[pred[0]]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36,255,12), 2)

        self.show_frame(frame)
        self.root.after(15, self.recognize_loop)

    def show_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((640, 480))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def exit_app(self):
        self.recognizing = False
        self.close_camera()
        self.root.destroy()


if __name__ == "__main__":
    root = Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
