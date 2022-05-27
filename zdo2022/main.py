import cv2
import numpy as np
import pandas as pd
from . import podpurne_funkce

class InstrumentTracker():
    def __init__(self):
        pass

    def predict(self, video_filename):
        """
        :param video_filename: name of the videofile
        :return: annnotations
        """

        #vytvoření anotace
        annotation = {
            "filename": [],
            "frame_id": [],
            "object_id": [],
            "x_px": [],  # x pozice obarvených hrotů v pixelech
            "y_px": [],  # y pozice obarvených hrotů v pixelech
            "annotation_timestamp": [],
        }

        #nacteni videa
        cap = cv2.VideoCapture(video_filename)

        #parametr pro vizualizaci
        #0 nevizualizovat, 1 vizualizovat
        vizualizovat = 1

        #pocitadlo snimku
        pocitadlo = 0

        while (1):
            #celkovy pocet snimku
            amount_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            #ukonceni nacitani jednotlivych snimku videa pokud je splnena podminka
            if pocitadlo == amount_of_frames:
                break

            pocitadlo += 1

            #pouze jeden sledovany nastroj a to needle holder
            objekt = 0

            #zapis do anotaci
            annotation['filename'].append(video_filename)
            annotation['frame_id'].append(pocitadlo)
            annotation['object_id'].append(objekt)

            #nacteni jednotliveho snimku videa
            ret, frame = cap.read()

            #vlozeni obdelniku v miste hrotu nuzek
            frame = cv2.rectangle(frame, (1307, 104), (1660, 327), (0, 0, 0), -1)

            #prevedeni snimku z BGR do HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            #nastaveni rozsahu cervene barvy
            lower_red = np.array([140, 150, 90])
            upper_red = np.array([180, 255, 255])

            #vytvoreni binarniho obrazku z pouze vybrane cervene barvy
            mask = cv2.inRange(hsv, lower_red, upper_red)

            #strukturni element
            kernel = np.ones((50, 50), np.uint8)

            #morfologicka operace uzavreni
            uzavreni = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # uzavreni

            #vytvoreni kontur
            contours, hierarchy = cv2.findContours(uzavreni, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            #vykresleni vsech kontur
            for c in contours:
                 cv2.drawContours(frame, [c], -1, (0, 255, 0), thickness=-1)

            #nastaveni souradnic pro pripad nenalezeni nastroje
            cx = -1
            cy = -1

            for c in contours:
                #vyber pouze takovych kontur, jejichz plocha je vetsi nez
                if cv2.contourArea(c) > 750:
                    #vytvoreni konvexniho obalu pouze te nejvetsi kontury z kontur vybranych
                    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
                    #vykresleni konvexniho obalu pouze nejvetsi kontury
                    cv2.drawContours(frame, [hull], -1, (255, 0, 0), -1)
                    #urceni centralniho momentu pro konvexni obal
                    M = cv2.moments(hull)
                    #urceni souradnic centroidu
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        cx, cy = 0, 0
                    #vykresleni centroidu
                    cv2.circle(frame, (cx, cy), 20, (255, 255, 255), -1)

            #zapis do anotace
            annotation['x_px'].append(cx)
            annotation['y_px'].append(cy)
            annotation['annotation_timestamp'].append(None)
            #vizualizace
            if vizualizovat == 1:
                imS2 = cv2.resize(frame, (800, 600))
                cv2.imshow("ConvexHull", imS2)
                cv2.waitKey(1)

        cap.release()
        cv2.destroyAllWindows()

        return annotation
        #return (print(pd.DataFrame(annotation)))


