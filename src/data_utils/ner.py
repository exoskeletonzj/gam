from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import json
import requests
import pandas
import math
from fastcoref import FCoref
from fastcoref import spacy_component
import spacy
API_URL = "https://api-inference.huggingface.co/models/dslim/bert-base-NER"
headers = {"Authorization": "Bearer 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"}
headers_token=['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','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','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','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','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']
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()
flag=0
def change_entity(type,entity,start_token):
    global flag
    if type=='train':
        if entity['word']=='. Keith Kellogg' :
            entity['word']='J. Keith Kellogg'
            entity['start']=98
        elif entity['word'] == 'N' and entity['end']==66:
            entity['word'] = 'U.N.'
            entity['start'] = 63
            entity['end']=67
        elif entity['word'] == 'BrucePurple':
            entity['word'] = '@BrucePurple'
            entity['start'] = 17
            entity['end']=29
        elif entity['word'] == 'Falsely':
            entity['word'] = '9.Falsely'
            entity['start'] = 0
        elif entity['word'] == 'C' and entity['end'] == 200:
            entity['word'] = 'R - S.C.'
            entity['start'] = 193
            entity['end'] = 201
        elif entity['word'] == 'S.' and entity['end'] == 76:
            entity['word'] = 'U.S.-allied'
            entity['start'] = 72
            entity['end'] = 83
        elif entity['word'] == 'CNN' and entity['end'] == 140:
            entity['word'] = '@CNN'
            entity['start'] = 136
        elif entity['word'] == 'Russian' and entity['end'] == 80 and entity['index']!= 11 and entity['index']!= 15 and entity['index']!= 13 and entity['index']!= 16 and entity['index']!= 14:
            entity['word'] = 'U.S.-Russian'
            entity['start'] = 68
        elif entity['word'] == 'Jack Ruby' and entity['end'] == 188:
            entity['word'] = 'prison—“Jack Ruby'
            entity['start'] = 171
        elif entity['word'] == 'Colombia Free Trade Agreement' and entity['end'] == 84:
            entity['word'] = 'U.S.-Colombia Free Trade Agreement'
            entity['start'] = 50
        elif entity['word'] == 'Russian' and entity['end'] == 232 and entity['index']!= 47 and entity['index']!= 49and entity['index']!= 48:
            entity['word'] = 'U.S.-Russian'
            entity['start'] = 220
        elif entity['word'] == 'India' and entity['end'] == 53:
            entity['word'] = 'U.S.-India'
            entity['start'] = 43
        elif entity['word'] == 'SpaceX' and entity['end'] == 99:
            entity['word'] = '@SpaceX'
            entity['start'] = 92
        elif entity['word'] == 'Russia' and entity['end'] == 54 and entity['index']==14:
            entity['word'] = 'U.S.-Russia'
            entity['start'] = 43
        elif entity['word'] == 'Soros' and entity['end'] == 336:
            entity['word'] = '“Soros'
            entity['start'] = 330
        elif entity['word'] == 'DavidCornDC' and entity['end'] == 27:
            entity['word'] = '@DavidCornDC'
            entity['start'] = 15
        elif entity['word'] == 'DavidCornDC' and entity['end'] == 12:
            entity['word'] = '@DavidCornDC'
            entity['start'] = 0
        elif entity['word'] == 'Bar' and entity['end'] == 136:
            entity['word'] = '@BarakRavid'
            entity['start'] = 132
            entity['end'] = 143
        elif entity['word'] == 'Tracinski' and entity['end'] == 55:
            entity['word'] = '@Tracinski'
            entity['start'] = 45
        elif entity['word'] == 'DerekJM' and entity['end'] == 32:
            entity['word'] = '@DerekJMurray'
            entity['start'] = 24
            entity['end'] = 37
        elif entity['word'] == 'StoneColdTru' and entity['end'] == 131:
            entity['word'] = '@StoneColdTruth'
            entity['start'] = 118
            entity['end'] = 133
        elif entity['word'] == 'RogerJS' and entity['end'] == 158:
            entity['word'] = '@RogerJStoneJr'
            entity['start'] = 150
            entity['end'] = 164
        elif entity['word'] == 'Jeff Jacob' and entity['end'] == 51:
            entity['word'] = '@Jeff Jacob'
            entity['start'] = 40
        elif entity['word'] == 'KingJ' and entity['end'] == 8:
            entity['word'] = '@KingJames'
            entity['start'] = 2
            entity['end'] = 12
        elif entity['word'] == 'HillaryClinton' and entity['end'] == 98:
            entity['word'] = '@HillaryClinton'
            entity['start'] = 83
        elif entity['word'] == 'AprilDR' and entity['end'] == 145:
            entity['word'] = '@AprilDRyan'
            entity['start'] = 137
            entity['end'] = 148
        elif entity['word'] == 'AlanGrey' and entity['end'] == 163:
            entity['word'] = '@AlanGreyProject'
            entity['start'] = 154
            entity['end'] = 170
        elif entity['word'] == '’ Brien' and entity['end'] == 7:
            entity['word'] = 'O’Brien'
            entity['start'] = 0
            entity['end'] = 6
        elif entity['word'] == 'Russia' and entity['end'] == 236:
            entity['word'] = 'U.S.–Russia'
            entity['start'] = 225
        elif entity['word'] == 'JoeTrip' and entity['end'] == 47:
            entity['word'] = '@JoeTrippi'
            entity['start'] = 39
            entity['end'] = 49
        elif entity['word'] == 'Colombian' and entity['end'] == 168:
            entity['word'] = '-Colombian'
            entity['start'] = 158
            entity['end'] = 168
        elif entity['word'] == 'Saudi' and entity['end'] == 167:
            entity['word'] = 'U.S.-Saudi'
            entity['start'] = 157
            entity['end'] = 167
        elif entity['word'] == 'PaulManafort' and entity['end'] == 33:
            entity['word'] = '@PaulManafort'
            entity['start'] = 20
            entity['end'] = 33
        elif entity['word'] == 'Russian' and entity['end'] == 16 and entity['index']==7:
            entity['word'] = 'U.S.–Russian'
            entity['start'] = 4
        elif entity['word'] == 'Russian' and entity['end'] == 115 and entity['index']==25:
            entity['word'] = 'U.S.–Russian'
            entity['start'] = 103
        elif entity['word'] == 'Iranian' and entity['end'] == 24 and entity['index']==8 and math.fabs(entity['score']-0.99946797)>0.01:
            entity['word'] = 'U.S./Iranian'
            entity['start'] = 12
        elif entity['word'] == 'KenanRah' and entity['end'] == 51 and entity['index']==28:
            entity['word'] = '@KenanRahmani'
            entity['start'] = 42
            entity['end'] = 55
        elif entity['word'] == 'IngrahamA' and entity['end'] == 29 and entity['index']==8:
            entity['word'] = '@IngrahamAngle'
            entity['start'] = 19
            entity['end'] = 33
        elif entity['word'] == 'DavidMDru' and entity['end'] == 31 :
            entity['word'] = '@DavidMDrucker'
            entity['start'] = 21
            entity['end'] = 35
        elif entity['word'] == 'BernieSanders' and entity['end'] == 32 :
            entity['word'] = '@BernieSanders'
            entity['start'] = 18
        elif entity['word'] == 'Morocco' and entity['end'] == 890:
            entity['word'] = 'U.S.-Morocco'
            entity['start'] = 878
        elif entity['word'] == 'Hillary' and entity['end'] == 358:
            entity['word'] = 'MSM.Hillary'
            entity['start'] = 347
        elif entity['word'] == 'Europe' and entity['end'] == 102 and entity['index'] != 20:
            entity['word'] = 'U.S.-Europe'
            entity['start'] = 91
        elif entity['word'] == 'Brien' and entity['end'] == 184:
            entity['word'] = "O'Brien"
            entity['start'] = 177
        elif entity['word'] == 'Brien' and entity['end'] == 146:
            entity['word'] = "O'Brien"
            entity['start'] = 139
        elif entity['word'] == 'NATO' and entity['end'] == 261:
            entity['word'] = 'U.S.-NATO'
            entity['start'] = 252
        elif entity['word'] == 'NickKristof' and entity['end'] == 33:
            entity['word'] = '@NickKristof'
            entity['start'] = 21
        elif entity['word'] == 'Frank Flannery' and entity['end'] == 23:
            entity['word'] = '-Frank Flannery'
            entity['start'] = 8
        elif entity['word'] == 'Giuseppe Donaldo Nicosia' and entity['end'] == 31:
            entity['word'] = '-Giuseppe Donaldo Nicosia'
            entity['start'] = 6
        elif entity['word'] =='Jehangir Soli Sorabjee' and entity['end'] == 29:
            entity['word'] = '-Jehangir Soli Sorabjee'
            entity['start'] = 6
        elif entity['word'] =='. K . P . Salve' and entity['end'] == 199:
            entity['word'] = 'N. K . P . Salve'
            entity['start'] = 185
        elif entity['word'] =='Turkish' and entity['end'] == 76:
            entity['word'] = 'U.S.-Turkish'
            entity['start'] = 64
        elif entity['word'] =='MFaul' and entity['end'] == 26:
            entity['word'] = '@McFaul'
            entity['start'] = 19
        elif entity['word'] =='JohnKasich' and entity['end'] == 27:
            entity['word'] = '@JohnKasich'
            entity['start'] = 16
        elif entity['word'] =='DonaldTrumpLA' and entity['end'] == 52:
            entity['word'] = '@DonaldTrumpLA'
            entity['start'] = 38
        elif entity['word'] =='’ Brien' and entity['end'] == 43:
            entity['word'] = 'O’Brien'
            entity['start'] = 36
        elif entity['word'] =='Saudi' and entity['end'] == 25:
            entity['word'] = 'U.S.-Saudi'
            entity['start'] = 15
        elif entity['word'] =='AlanGray' and entity['end'] == 39:
            entity['word'] = '@AlanGrayson'
            entity['start'] = 30
            entity['end'] = 42
        elif entity['word'] =='’ Neill' and entity['end'] == 46:
            entity['word'] = 'O’Neill'
            entity['start'] = 39
        elif entity['word'] =='DavidBegna' and entity['end'] ==174:
            entity['word'] = '@DavidBegnaud'
            entity['start'] = 163
            entity['end'] = 176
        elif entity['word'] =='Russia' and entity['end'] ==128 and entity['index'] ==34:
            entity['word'] = 'U.S.-Russia'
            entity['start'] = 117
            entity['end'] = 128
        elif entity['word'] =='Israel' and entity['end'] ==248 and entity['index'] !=50:
            entity['word'] = 'U.S.-Israel'
            entity['start'] = 237
        elif entity['word'] =='. Ann Selzer' and entity['end'] ==124:
            entity['word'] = 'J.Ann Selzer'
            entity['start'] = 111
        elif entity['word'] =='Israel' and entity['end'] ==219:
            entity['word'] = 'U.S.-Israel'
            entity['start'] = 208
        elif entity['word'] =='Israel' and entity['end'] ==131:
            entity['word'] = 'U.S.-Israel'
            entity['start'] = 120
        elif entity['word'] =='Russia' and entity['end'] ==212 and entity['index'] ==46:
            entity['word'] = 'U.S.-Russia'
            entity['start'] = 201
        elif entity['word'] =='CNNPolitics' and entity['end'] ==168:
            entity['word'] = '@CNNPolitics'
            entity['start'] = 156
        elif start_token !=28 and entity['word'] =='Israeli' and entity['end'] ==45 and entity['index'] ==9 and math.fabs(entity['score']-0.999669)<0.00001:
            entity['word'] = 'U.S.-Israeli'
            entity['start'] = 33
        elif entity['word'] =='Russia' and entity['end'] ==92 and entity['index'] ==23:
            entity['word'] = 'U.S.-Russia'
            entity['start'] = 80
        elif entity['word'] =='Russian' and entity['end'] ==92 and entity['index'] ==23:
            entity['word'] = 'U.S.-Russian'
            entity['start'] = 80
        elif entity['word'] =='Middle Eastern' and entity['end'] ==76 and entity['index'] ==16:
            entity['word'] = 'U.S.-Middle Eastern'
            entity['start'] = 57
        elif entity['word'] =='Saudi Arabia' and entity['end'] ==239 and entity['index'] ==47:
            entity['word'] = 'U.S.-Saudi Arabia'
            entity['start'] = 222
        elif entity['word'] == 'TimNaftali' and entity['end'] == 27 and entity['index'] == 8:
            entity['word'] = '@TimNaftali'
            entity['start'] = 16
        elif entity['word'] == 'Pope' and entity['end'] == 41 and entity['index'] == 22:
            entity['word'] = '@Popehat'
            entity['start'] = 36
            entity['end'] = 44
        elif entity['word'] == '. N . Security Council' and entity['end'] == 58 :
            entity['word'] ='U.N. Security Council'
            entity['start'] = 37
        elif entity['word'] == 'BernieSanders' and entity['end'] == 42 and entity['index'] == 12:
            entity['word'] = '@BernieSanders'
            entity['start'] = 28
        elif entity['word'] == 'ABCPolitics' and entity['end'] == 167 and entity['index'] == 53:
            entity['word'] = '@ABCPolitics'
            entity['start'] = 155
        elif entity['word'] == 'LizziePhelan' and entity['end'] == 113 and entity['index'] == 39:
            entity['word'] = '@LizziePhelan'
            entity['start'] = 100
        elif entity['word'] == 'LizziePhelan' and entity['end'] == 177 and entity['index'] == 55:
            entity['word'] = '@LizziePhelan'
            entity['start'] = 164
        elif entity['word'] == 'DebraM' and entity['end'] == 7 and entity['index'] == 2:
            entity['word'] = '@DebraMax'
            entity['start'] = 0
            entity['end'] = 9
        elif entity['word'] == 'MichaelRCap' and entity['end'] == 31 and entity['index'] == 7:
            entity['word'] = '@MichaelRCaputo'
            entity['start'] = 19
            entity['end'] = 34
        elif entity['word'] == 'McCormickPro' and entity['end'] == 34 and entity['index'] == 8:
            entity['word'] = '@McCormickProf'
            entity['start'] = 21
            entity['end'] = 35
        elif entity['word'] =='Russian' and entity['end'] ==164 and entity['index'] ==35:
            entity['word'] = 'U.S.-Russian'
            entity['start'] = 152
        elif entity['word'] =='MiddleEastE' and entity['end'] ==129 and entity['index'] ==54:
            entity['word'] = '@MiddleEastEye'
            entity['start'] = 117
            entity['end'] = 131
        elif entity['word'] =='RyanLiz' and entity['end'] ==23 and entity['index'] ==7:
            entity['word'] = '@RyanLizza'
            entity['start'] = 15
            entity['end'] = 25
        elif entity['word'] =='NATO' and entity['end'] ==184 and entity['index'] ==43:
            entity['word'] = 'U.S./NATO'
            entity['start'] = 175
        elif entity['word'] =='DanaBashCNN' and entity['end'] ==161 and entity['index'] ==44:
            entity['word'] = '@DanaBashCNN'
            entity['start'] = 149
        elif entity['word'] =='Mexico' and entity['end'] ==84 and entity['index'] ==17:
            entity['word'] = 'U.S.-Mexico'
            entity['start'] = 73
        elif entity['word'] =='AIBA Boxing' and entity['end'] ==171 and entity['index'] ==50:
            entity['word'] = '@AIBA_Boxing'
            entity['start'] = 159
        elif entity['word'] =='EddieHearn' and entity['end'] ==27 and entity['index'] ==7:
            entity['word'] = '@EddieHearn'
            entity['start'] = 16
        elif entity['word'] =='Russia' and entity['end'] ==63 and entity['index'] ==18:
            entity['word'] = 'U.S.-Russia'
            entity['start'] = 52
        elif entity['word'] =='Israeli' and entity['end'] ==56 and entity['index'] ==15:
            entity['word'] = 'U.S.-Israeli'
            entity['start'] = 44
        elif entity['word'] =='Philippine' and entity['end'] ==52 and entity['index'] ==15:
            entity['word'] = 'U.S.-Philippine'
            entity['start'] = 37
        elif entity['word'] =='. Scott Applewhit' and entity['end'] ==20 and entity['index'] ==3:
            entity['word'] = 'J. Scott Applewhit'
            entity['start'] = 2
        elif entity['word'] =='KhalilNoor' and entity['end'] ==158 and entity['index'] ==35:
            entity['word'] = '@KhalilNoori'
            entity['start'] = 147
            entity['end'] = 159
        elif entity['word'] =='ComradZampoli' and entity['end'] ==117 and entity['index'] ==20:
            entity['word'] = '@ComradZampolit'
            entity['start'] = 103
            entity['end'] = 118
        elif entity['word'] =='Chinese' and entity['end'] ==34 and entity['index'] ==10:
            entity['word'] = 'U.S.-Chinese'
            entity['start'] = 22
            entity['end'] = 34
        elif entity['word'] =='Russia' and entity['end'] ==102 and entity['index'] ==23 and math.fabs(entity['score']-0.9995211)<0.0001:
            if flag==0:
                return entity
                flag=1
            entity['word'] = 'U.S.-Russia'
            entity['start'] = 91
            flag=1
        elif entity['word'] =='HillaryClinton' and entity['end'] ==373 and entity['index'] ==88:
            entity['word'] = '@HillaryClinton'
            entity['start'] = 358
            entity['end'] = 373
        elif entity['word'] =='ABC' and entity['end'] ==68 and entity['index'] ==41:
            entity['word'] = '@ABC'
            entity['start'] = 64
        elif entity['word'] =='International Business Times' and entity['end'] ==146 and entity['index'] ==35:
            entity['word'] = '-International Business Times'
            entity['start'] = 117
        elif entity['word'] == 'WhiteGenocideTM' and entity['end'] == 48 and entity['index'] == 11:
            entity['word'] = '@WhiteGenocideTM'
            entity['start'] = 32
        elif entity['word'] == '. Mitchell Palmer' and entity['end'] == 198 and entity['index'] ==30:
            entity['word'] = 'A.Mitchell Palmer'
            entity['start'] = 180
        elif entity['word'] == 'Russian' and entity['end'] == 232 and entity['index'] ==49 and math.fabs(entity['score']-0.99974465)>0.000001:
            entity['word'] = 'U.S.-Russian'
            entity['start'] = 220
        elif entity['word'] == 'Dutch' and entity['end'] == 41 and entity['index'] ==14:
            entity['word'] = 'U.S.-Dutch'
            entity['start'] = 31
        elif entity['word'] == 'TanyaCoop' and entity['end'] == 65 and entity['index'] ==19:
            entity['word'] = '@TanyaCooper'
            entity['start'] = 55
            entity['end'] = 67
        elif entity['word'] == 'Alyona Pritula' and entity['end'] == 15 and entity['index'] ==1:
            entity['word'] = 'Alyona Pritula'
            entity['start'] = 0
        elif entity['word'] == 'LeslieSanche' and entity['end'] == 32 and entity['index'] ==6:
            entity['word'] = '@LeslieSanchez'
            entity['start'] = 19
            entity['end'] = 33
        elif entity['word'] == 'CahnEmily' and entity['end'] == 168 and entity['index'] ==42:
            entity['word'] = '@CahnEmily'
            entity['start'] = 158
            entity['end'] = 168
        elif entity['word'] == 'TaraSetmayer' and entity['end'] == 40 and entity['index'] ==10:
            entity['word'] = '@TaraSetmayer'
            entity['start'] = 27
        elif entity['word'] == 'Japan' and entity['end'] == 130 and entity['index'] ==31:
            entity['word'] = 'U.S.-Japan'
            entity['start'] = 120
        elif entity['word'] == '’ Brien' and entity['end'] == 193 and entity['index'] ==39:
            entity['word'] = 'O’Brien'
            entity['start'] = 186
        elif entity['word'] == '’ Brien' and entity['end'] == 138 and entity['index'] ==29:
            entity['word'] = 'O’Brien'
            entity['start'] = 131
        elif entity['word'] == 'Iranian' and entity['end'] == 24 and entity['index'] == 8 and math.fabs(
                entity['score'] - 0.99879056) < 0.0001:
            entity['word'] = 'U.S./Iranian'
            entity['start'] = 12
        elif entity['word'] == 'NatShu' and entity['end'] == 48 and entity['index'] == 25 :
            entity['word'] = '@NatShupe'
            entity['start'] = 41
            entity['end'] = 50
        elif entity['word'] == 'NatShu' and entity['end'] == 48 and entity['index'] == 25 :
            entity['word'] = '@NatShupe'
            entity['start'] = 41
            entity['end'] = 50
        elif entity['word'] == '. Matheswaran' and entity['end'] == 128 and entity['index'] == 24 :
            entity['word'] = 'M. Matheswaran'
            entity['start'] = 114
            entity['end'] = 128
        elif  entity['word'] =='Israeli' and entity['end'] ==45 and entity['index'] ==9 and math.fabs(entity['score']-0.99623257)<0.00001:
            entity['word'] = 'U.S.-Israeli'
            entity['start'] = 33
        elif  entity['word'] =='Israeli' and entity['end'] ==45 and entity['index'] ==16 and math.fabs(entity['score']-0.99623257)<0.00001:
            entity['word'] = 'U.S.-Israeli'
            entity['start'] = 33
        elif entity['word'] =='Russia' and entity['end'] ==102 and entity['index'] ==23 and math.fabs(entity['score']-0.6596362)<0.0001:
            entity['word'] = 'U.S.-Russia'
            entity['start'] = 91
    elif type == 'dev':
        if  entity['word'] =='Schwarzene' and entity['end'] ==39 and entity['index'] ==7 :
            entity['word'] = '@Schwarzenegger'
            entity['start'] = 28
            entity['end'] = 43
        elif  entity['word'] =='ScottWalk' and entity['end'] ==154 and entity['index'] ==56 :
            entity['word'] = '@ScottWalker'
            entity['start'] = 144
            entity['end'] = 156
        elif  entity['word'] =='SouthwestAir' and entity['end'] ==53 and entity['index'] ==14 :
            entity['word'] = '@SouthwestAir'
            entity['start'] = 40
            entity['end'] = 53
        elif entity['word']=='. Keith Kellogg' and entity['end'] ==114 and entity['index'] ==19:
            entity['word']='J. Keith Kellogg'
            entity['start']=98
        elif entity['word']=='Gary Bass' and entity['end'] ==114 and entity['index'] ==40:
            entity['word']='@Gary__Bass'
            entity['start']=103
        elif entity['word']=='Russia' and entity['end'] ==211 and entity['index'] ==47:
            entity['word']='U.S.-Russia'
            entity['start']=200
        elif entity['word'] == 'HillaryClinton' and entity['end'] == 35 and entity['index'] == 6:
            entity['word'] = '@HillaryClinton'
            entity['start'] = 20
        elif entity['word'] == '. Steven Fish' and entity['end'] == 99 and entity['index'] == 17:
            entity['word'] = 'M. Steven Fish'
            entity['start'] = 85
        elif entity['word'] == 'BarzanSadiq' and entity['end'] == 29 and entity['index'] == 8:
            entity['word'] = '@BarzanSadiq'
            entity['start'] = 17
        elif entity['word'] == 'JustinWolfers' and entity['end'] == 33 and entity['index'] == 7:
            entity['word'] = '@JustinWolfers'
            entity['start'] = 19
        elif entity['word'] == 'Stephen' and entity['end'] == 18 and entity['index'] == 4:
            entity['word'] = '@StephenMeister'
            entity['start'] = 10
            entity['end'] = 25
        elif entity['word'] == 'Israeli' and entity['end'] == 58 and entity['index'] == 16:
            entity['word'] = 'U.S.-Israeli'
            entity['start'] = 46
    elif type == 'test':
        if entity['word'] == 'Mexico' and entity['end'] == 115 and entity['index'] == 26:
            entity['word'] = 'U.S.-Mexico'
            entity['start'] = 104
        elif entity['word'] == 'Saudi' and entity['end'] == 59 and entity['index'] == 16:
            entity['word'] = 'U.S.-Saudi'
            entity['start'] = 49
        elif entity['word'] == 'Sweden' and entity['end'] == 23 and entity['index'] == 6:
            entity['word'] = '@Sweden'
            entity['start'] = 16
        elif entity['word'] == 'TheLocalSweden' and entity['end'] == 104 and entity['index'] == 36:
            entity['word'] = '@TheLocalSweden'
            entity['start'] = 89
        elif entity['word'] == 'exxonmobil' and entity['end'] == 17 and entity['index'] == 4:
            entity['word'] = '@exxonmobil'
            entity['start'] = 6
        elif entity['word'] == 'HouseScience Committee' and entity['end'] == 62 and entity['index'] == 14:
            entity['word'] = '@HouseScience Committee'
            entity['start'] = 39
        elif entity['word'] == 'MassAGO' and entity['end'] == 139 and entity['index'] == 40:
            entity['word'] = '@MassAGO'
            entity['start'] = 131
        elif entity['word'] == '. A . Goodman' and entity['end'] == 56 and entity['index'] == 9:
            entity['word'] = 'H. A . Goodman'
            entity['start'] = 43
        elif entity['word'] == 'Bernstein' and entity['end'] == 147 and entity['index'] == 43:
            entity['word'] = '@Bernstein'
            entity['start'] = 137
        elif entity['word'] == '. K . Takkar' and entity['end'] == 19 and entity['index'] == 5:
            entity['word'] = 'R.K . Takkar'
            entity['start'] = 8
    return entity

def find_start_end(sentence,entity,start_token):
    entity=change_entity('dev',entity,start_token)
    start_sent_id = 0
    start_id = 0
    while start_id != entity['start'] and start_sent_id<len(sentence):
        start_id+=len(sentence[start_sent_id])+1
        start_token+=1
        start_sent_id+=1
    if  start_sent_id>=len(sentence):
        return -1,-1
    end_id=start_id+len(sentence[start_sent_id])
    end_token=start_token
    start_sent_id+=1
    while end_id != entity['end'] and start_sent_id<len(sentence):
        end_id+=len(sentence[start_sent_id])+1
        end_token+=1
        start_sent_id+=1

    return start_token,end_token+1

def modify(ner):
    i=0
    while i<len(ner)-1:
        if len(ner[i+1]['word'])>2 and ner[i+1]['word'][:2]=='##':
            ner[i]['word']+=ner[i+1]['word'][2:]
            ner[i]['end'] = ner[i + 1]['end']
            ner.pop(i+1)
            i-=1
        i+=1
    i = 0
    while i<len(ner)-1:
        if  ner[i+1]['entity'][0]=='I':
            ner[i]['word']+= ' '+ner[i+1]['word']
            ner[i]['end'] = ner[i + 1]['end']
            ner.pop(i+1)
            i-=1
        i+=1
    return ner
def get_trigger_sent(sentence,id):
    token_id=0
    for i in range(len(sentence)):
        token_id+=len(sentence[i])
        if token_id> id:
            return i
    return -1
def check(entity):
    if entity['word']=='Snowden' and entity['start']==38:
        return True
    elif entity['word']=='Aleed Talal' and entity['start']==140:
        return True
    elif entity['word']=='Arabs' and entity['start']==165:
        return True
    elif entity['word']=='Apple' and entity['start']==76:
        return True
    elif entity['word']=='Afghanistan' and entity['start']==61:
        return True
    elif entity['word']=='Muslim' and entity['start']==90:
        return True
    elif entity['word']=='Rep' and entity['start']==1:
        return True
    elif entity['word']=='U . K .' and entity['start']==23:
        return True
    elif entity['word']=='H' and entity['start']==1:
        return True
    elif entity['word']=='Hillary Clinton' and entity['start']==233:
        return True
    elif entity['word']=='Saudi' and entity['start']==263:
        return True
    elif entity['word']=='Johnson' and entity['start']==105:
        return True
    elif entity['word']=='. O' and entity['start']==195:
        return True
    elif entity['word']=='. Key' and entity['start']==198:
        return True
    elif entity['word']=='Clinton' and entity['start']==109:
        return True
    elif entity['word']=='SA' and entity['start']==170:
        return True
    elif entity['word']=='Aleppo' and entity['start']==175:
        return True
    elif entity['word']=='TOO' and entity['start']==137:
        return True
    elif entity['word']=='Azerbaijan Crime' and entity['start']==163:
        return True
    elif entity['word']=='. N' and entity['start']==56:
        return True
    elif entity['word']=='CoWest' and entity['start']==17:
        return True
    elif entity['word']=='César Rosenthal' and entity['start']==10:
        return True
    elif entity['word']=='Jean Claude N ’ Da Ametchi' and entity['start']==13:
        return True
    elif entity['word']=='” Clinton' and entity['start']==3:
        return True
    elif entity['word']=='Russia' and entity['start']==102:
        return True
    elif entity['word']=='Belgian' and entity['start']==542:
        return True
    elif entity['word']=='. Charles McCullough' and entity['start']==76:
        return True
    elif entity['word']=='Abs NBC' and entity['start']==104:
        return True
    elif entity['word']=='Russia' and entity['start']==17 and entity['index']==10:
        return True
    elif entity['word']=='Wall Street' and entity['start']==161 and entity['index']==52:
        return True
    elif entity['word']=='freedom' and entity['start']==109 and entity['index']==28:
        return True
    elif entity['word']=='AIBA' and entity['start']==1 and entity['index']==2:
        return True
    elif entity['word']=='Al' and entity['start']==81 and entity['index']==18:
        return True
    elif entity['word']=='NovSP' and entity['start']==211 and entity['index']==48:
        return True
    elif entity['word']=='MHway' and entity['start']==12 and entity['index']==6:
        return True
    elif entity['word']=='The Wall Street Journal' and entity['start']==65 and entity['index']==17:
        return True
    elif entity['word']=='. Bush' and entity['start']==130 and entity['index']==30:
        return True
    elif entity['word'] == '’ Keef' and entity['end'] == 10 and entity['index'] ==4:
        return True
    elif entity['word'] == '’ Keef' and entity['end'] == 21 and entity['index'] ==7:
        return True
    elif entity['word'] == 'HillaryClintonDonalTrump' and entity['end'] == 133 and entity['index'] ==19:
        return True
    elif entity['word'] == 'Evgeny Kaveshniko' and entity['end'] == 77 and entity['index'] ==15:
        return True
    elif entity['word'] == '’' and entity['end'] == 10 and entity['index'] ==5:
        return True
    elif entity['word'] == 'WJ' and entity['end'] == 31 and entity['index'] ==10:
        return True
    elif entity['word'] == 'D Va' and entity['end'] == 121 and entity['index'] ==26:
        return True
    elif entity['word'] == 'Y' and entity['end'] == 108 and entity['index'] ==50:
        return True
    elif entity['word'] == 'Law' and entity['end'] == 83 and entity['index'] ==23:
        return True
    elif entity['word'] == 'S Muslim' and entity['end'] == 2039 and entity['index'] ==393:
        return True
    elif entity['word'] == '’' and entity['end'] == 360 and entity['index'] ==97:
        return True
    elif entity['word'] == 'Amos' and entity['end'] == 13 and entity['index'] ==3:
        return True
    elif entity['word'] == 'BBCA' and entity['end'] == 65 and entity['index'] ==25:
        return True
    elif entity['word'] == 'Lerner' and entity['end'] == 193 and entity['index'] ==42:
        return True
    elif entity['word'] == 'Foreign Press Association' and entity['end'] == 246 and entity['index'] ==47:
        return True
    elif entity['word'] == 'Foreign Affairs and Defense Committee' and entity['end'] == 238 and entity['index'] ==42:
        return True
    elif entity['word'] == 'IDF' and entity['end'] == 475 and entity['index'] ==97:
        return True
    return False
def get_entity(result,nlp):
    token_id = 1
    entity_mentions = []
    token = []
    sent_idx = 0
    start_tokenid = 0
    count=0
    for sentence_list in result['sentences']:
        input = {}
        sentence = ' '.join(sentence_list)
        token += sentence_list
        # print(sentence)
        input['inputs'] = sentence
        # input['parameters']={"aggregation_strategy": "nonesimple"},
        # ner_results = query(input)


        ner_results = nlp(sentence)

        ner_results = modify(ner_results)
        if ner_results=='error':
            headers['Authorization']=headers['Authorization'][:7]+headers_token[token_id]
            token_id+=1
        # print(ner_results)
        for entity in ner_results:
            if entity['entity'][0]=='I' or len(entity['word'])>2 and entity['word'][:2]=='##' or check(entity):
                continue
            entity_mention = {}
            entity_mention['id'] = result['doc_key']+'-T'+str(count)
            count+=1
            entity_mention['sent_idx'] = sent_idx
            entity_mention['text'] = entity['word']
            entity_mention['entity_type'] = entity['entity'][2:]
            start_token, end_token = find_start_end(sentence_list, entity, start_tokenid)
            if start_token==-1 and end_token ==-1:
                continue
            entity_mention['start'] = start_token
            entity_mention['end'] = end_token
            entity_mentions.append(entity_mention)
            # print(start_token,end_token)
        start_tokenid += len(sentence_list)
        sent_idx += 1

    return token,entity_mentions

def get_trigger(result,token,entity_mentions):
    event_mentions=[]
    evt_dict = {}
    event_id=0
    for event in result['evt_triggers']:
        event_metion = {}
        event_metion['event_type'] = event[2][0][0]
        trigger = {}
        trigger['start'] = event[0]
        trigger['end'] = event[1] + 1
        trigger['text'] = ''
        trigger['sent_idx'] = get_trigger_sent(result['sentences'], event[0])
        evt_dict[event[0]]=event_id
        for i in range(event[0], event[1] + 1):
            trigger['text'] += token[i]
            if i != event[1]:
                trigger['text'] += ' '
        event_metion['trigger']=trigger
        event_metion['arguments']=[]
        event_metion['id']=result['doc_key']+'-E'+str(event_id)
        event_mentions.append(event_metion)
        event_id+=1
        # print(trigger['text'])

    for link in result['gold_evt_links']:
        argument={}
        argument['role']=link[2][11:]
        argument['text']=''
        argument['start'] = link[1][0]
        argument['end'] = link[1][1]+1
        for i in range(link[1][0], link[1][1] + 1):
            argument['text'] += token[i]
            if i != link[1][1]:
                argument['text'] += ' '
        event_mentions[evt_dict[link[0][0]]]['arguments'].append(argument)

    return event_mentions
def get_sentences(result):
    sentences=[]
    for sent in result['sentences']:
        res_sent=[]
        for word in sent:
            res_word=[]
            res_word.append(word)
            res_sent.append(res_word)
        tem=[]
        tem.append(res_sent)
        tem.append(''.join(sent))
        sentences.append(tem)
    return sentences


def get_ner_database(source,target,nlp):

    with open(source, 'r') as reader, open(target, 'w') as writer:
        count=0
        for line in reader:
            # if count<600:
            #     count += 1
            #     continue
            result = json.loads(line)

            tokens,entity_mentions=get_entity(result,nlp)
            event_mentions=get_trigger(result,tokens,entity_mentions)
            sentences=get_sentences(result)
            # print(event_mentions)
            # get_agu_metion(entity_mentions,event_mentions,result)
            processed_ex = {
                'doc_id': result['doc_key'],
                'tokens':tokens,
                'sentences':sentences,
                'entity_mentions':entity_mentions,
                'event_mentions':event_mentions
            }
            # print(processed_ex)
            writer.write(json.dumps(processed_ex) + '\n')
            count+=1


def get_json_rams(source,target):
    data =pandas.read_csv(source)
    processed={}
    # print(data)
    with open(target, 'w') as writer:
        for i, line in data.iterrows():
            event_id=str(line[0])[:-13]
            processed[event_id]={}
            processed[event_id]['template']=str(line[1])[1:]
            processed[event_id]['roles']=[]
            processed[event_id]['role_types']=[]
        writer.write(json.dumps(processed))

def get_cor(source,target):
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("fastcoref")
    # o = 0
    with open(source, 'r') as reader, open(target, 'w') as writer:
        for line in reader:
            # o += 1
            # if o < 460:
            #     continue
            result = json.loads(line)
            text=' '.join(result['tokens'])
            start_id=0
            dict_metion={}
            count=0
            id=0

            for token in result['tokens']:
                if id + 1<len(result['entity_mentions']) and result['entity_mentions'][id]['end'] > result['entity_mentions'][id+1]['start']:
                    id+=1
                # if count==133 and result['doc_id']=='nw_RC021233e0ea88f2f267dc69fcfa1207c4bb50259fee285ca31c79cab2':
                #     print(1)
                if id == len(result['entity_mentions']):
                    break
                if count == result['entity_mentions'][id]['start']:
                    result['entity_mentions'][id]['stringid'] = (start_id,start_id + len(token))
                    if result['entity_mentions'][id]['end'] ==result['entity_mentions'][id]['start']+1:
                        dict_metion[(start_id,start_id + len(token))]=result['entity_mentions'][id]['id']
                        id+=1
                elif count < result['entity_mentions'][id]['start']:
                    count+=1
                    start_id += len(token)
                    start_id += 1
                    continue
                elif count == result['entity_mentions'][id]['end']:
                    id+=1
                    if id < len(result['entity_mentions']) and count == result['entity_mentions'][id]['start']:
                        result['entity_mentions'][id]['stringid'] = (start_id, start_id + len(token))
                        if result['entity_mentions'][id]['end'] == result['entity_mentions'][id]['start'] + 1:
                            dict_metion[(start_id, start_id + len(token))] = result['entity_mentions'][id]['id']
                            id += 1
                elif count > result['entity_mentions'][id]['end']:
                    id+=1
                else :
                    result['entity_mentions'][id]['stringid'] = (result['entity_mentions'][id]['stringid'][0], result['entity_mentions'][id]['stringid'][1] +len(token))
                    if count == result['entity_mentions'][id]['start'] == count -1:
                        dict_metion[(result['entity_mentions'][id]['stringid'][0], result['entity_mentions'][id]['stringid'][1] +len(token))] = result['entity_mentions'][id]['id']
                        id +=1
                start_id += len(token)
                start_id+=1
                count += 1
            doc = nlp(text)
            clusters=[]
            for coref in doc._.coref_clusters:
                mini_clusters = []
                for i in coref:
                    if i in dict_metion:
                        mini_clusters.append(dict_metion[i])
                if len(mini_clusters)!=0 :
                    clusters.append(mini_clusters)
            processed_ex = {
                'doc_key': result['doc_id'],
                'clusters': clusters,
            }
            writer.write(json.dumps(processed_ex) + '\n')


if __name__ == "__main__":

    target = '../../data/rams/train_ner.jsonlines'
    target_dev = '../../data/rams/dev_ner.jsonlines'
    target_test = '../../data/rams/test_ner.jsonlines'




    target_coref_dev = '../../data/rams/coref/dev.jsonlines'
    target_coref_test = '../../data/rams/coref/test.jsonlines'
    target_coref_train='../../data/rams/coref/train.jsonlines'
    get_cor(target,target_coref_train)
    get_cor(target_dev, target_coref_dev)
    get_cor(target_test, target_coref_test)

