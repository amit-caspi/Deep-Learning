
"""
Created on Sat Nov 21 14:10:31 2020

@author: Topaz Ben Atar, Amit Caspi, Roni Peri, Maya Yosef, Ron Amado.

"""

import os 
import shutil
import glob  
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def makealibrary(): 
    """
    paramters: none.
    
    Input path and library name. If the library does not exist, 
    the function creates a new library.
    If the input is 'ENTER', the defult path is desktop.
    In the library creates two libraries- "train" and "test". 
    
    Returns: none.
    """
    path=input("Please enter a path: ")
    library_name=input("Please enter a library name: ")
    if path== "":  #במקרה של אנטר
       path=r"C:\Users\Topaz\Desktop" #נתיב לשולחן הבית בהתאם למחשב
    path1 = os.path.join(path, library_name) #יצירת נתיב לספרייה 
    if not os.path.isdir(path1): #במקרה בו הנתיב לא קיים 
        print ("Does not exist, creating new library named "+ library_name)
        os.mkdir(path1) #יצירת הנתיב לספרייה
    else:
        print ("Already exists")
    global trainway, testway
    trainway= os.path.join(path1, "train") #הנתיב לספרייה train 
    testway= os.path.join(path1, "test") #הנתיב לספרייה test
    os.mkdir(trainway) #יצירת הנתיב לtrain 
    os.mkdir(testway) #יצירת הנתיב לtest 


def imagefunction(dest, jpgpath):
    """
    Parameters: path (train or test) according to user choice, which is our dest
    and a string which describe the path of a library which contains jpg images.
    
    If the dest is train, the function moves 70% of the images from the images 
    library into train library.
    If the dest is test, the function moves all of the images from the images 
    library into test library
    
    Returns: none.
    """
    source = jpgpath #ספריית המקור שלנו, אשר מכילה את כל התמונות
    if dest== trainway: 
        numofimages= 0.7*len(os.listdir(jpgpath)) #משתנה המכיל מספר המהווה 70 אחוז מכל התמונות
        counter=0 #משתנה הגדל עד אשר שווה למספר המהווה 70 אחוז מהתמונות
        for jpgfile in glob.iglob(os.path.join(source, "*.jpg")):
            if (counter != numofimages): 
                shutil.move(jpgfile, dest)
                counter= counter + 1
    if dest==testway : #יכניס את כל התמונות 
        for jpgfile in glob.iglob(os.path.join(source, "*.jpg")):
            shutil.move(jpgfile, dest)


def showfiles(currentpath):
    """
    paramters: currentpath- trainway or testway 
    print the image's names and shows the image in plots.
    Returns: none.
    """
    filesnames= os.listdir(currentpath) #רשימה שתכיל את שמות התמונות בנתיב הנבחר 
    print ("The pictures names are: ")
    for filename in filesnames:    
       image = mpimg.imread(os.path.join(currentpath,filename)) #קריאת מידע התמונה
       plt.axis("off") 
       plt.imshow(image)
       plt.show()
       print (filename)
    print ("The images show above in plots.")
      
        
def main():
    makealibrary()
    jpgpath= r"C:\Users\Topaz\Pictures\animls" #נתיב לספריית התמונות מסוג jpg
    whichone= input ("choose train or test: ")
    if (whichone== "train"):
        imagefunction(trainway, jpgpath)
        showfiles(trainway)
    if (whichone== "test"):
        imagefunction(testway, jpgpath)
        showfiles(testway)
    
if __name__ == "__main__": 
    main()
