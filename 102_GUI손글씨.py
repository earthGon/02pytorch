from tkinter import *
from PIL import Image, ImageDraw
import torch
import os
import numpy as np

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 10),
        )
    def forward(self,x):
        y = self.model(x)
        return y

model = MyModel()
model.load_state_dict(torch.load("myMNISTmodel.pth"))

class drawMNIST(Tk):  ##Tk가 아니라 Frame을 받았네?
    def __init__(self):
        super().__init__()  
        self.title="MNIST"
        self.geometry("280x350")

        self.button = Button(
            # master, text=name, command=self.update_display, image=self.img_preview,
            self,
            text="reset",
            command=self.reset,
        )
        # self.label = Label(master, text=name, bg="white")
        self.label = Label(self, text="", bg="white")

        self.canvas = Canvas(self,width=280, height=280, bg="black")
        self.canvas.bind("<B1-Motion>", self.draw) 
        self.canvas.bind("<ButtonRelease-1>", self.inference) 

        self.canvas.pack(side=TOP, fill=BOTH, expand=False)
        self.label.pack(side=TOP, fill=BOTH, expand=False)
        self.button.pack(side=TOP, fill=BOTH, expand=False)

        
        self.img_src = Image.new("RGB",(280,280), (0, 0, 0))
        self.image_layer = ImageDraw.Draw(self.img_src, "RGB")

        self.old_x =None
        self.old_y =None
        
    def draw(self,e):
        if (self.old_x):
            self.canvas.create_line(self.old_x,self.old_y,e.x,e.y, width=3, fill="white")
            self.image_layer.line(
                [self.old_x, self.old_y, e.x, e.y],width=3, fill="white"
                )
        self.old_x = e.x
        self.old_y = e.y
        
    def inference(self,e):
        self.old_x=None
        self.old_y=None
        # image를 numpy로 바꾸기
        # numpy.clip (0,1) , shape 조절
        # numpy /255, dtype =np.float32
        # torch로 변환
        
        img = self.img_src.resize((28, 28))
        img = np.array(img)
        img = np.where(img > 0, 255, 0) 
        
        #디버깅용
        # image.fromarray(np.astype(np.uint8)).save("my_png.jpg")
        
        # 차원 줄이기 (흑백으로)
        # tensor만들기 (1,28*28)
        # model 넣기
        img = img[:,:,0]
        img = img/255
        data =torch.tensor(img,dtype=torch.float32).reshape(-1,28*28)
        result = torch.argmax(model(data),dim=1)[0]

        self.label["text"] = f"tensor ({result})"

    def reset(self):
        # img_src를 black으로 변환 어떻게 하지
        self.image_layer.rectangle((0, 0, 280, 280), fill=(0, 0, 0))
        self.canvas.delete("all")       
        self.label["text"] = ""


if __name__=='__main__':
    
    drawMNIST().mainloop()