# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 14:39:58 2018

@author: julien
"""

import pickle
from NN import *
import numpy as np

class Game(object):
    def __init__(self, first_gamer='HU', IA='net.pkl'):
        '''
        vide : -
        IA : X
        humain : O
        '''
        self.first_gamer=first_gamer
        with open(IA, 'rb') as input:
            self.net = pickle.load(input)
        self.X=[]
        self.Y=[]
        self.pos = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
        
    def run(self) :
        self.board = [['-','-','-'],
                      ['-','-','-'],
                      ['-','-','-']]
        self.X=[]
        self.Y=[]
        gamer = self.first_gamer
        self.printBoard(self.board)
        while True :
            if gamer == 'HU' : #Défini si c'est le tour de l'IA ou de l'utilisateur
                #Demande une position en coordonnés à l'utilisateur
                err = True
                inputs = self.boardConv()
                outputs = self.net.feedforward(inputs)
                while err == True :
                    x = min(max(int(input('x (0-2) : ')),0),2)
                    y = min(max(int(input('y (0-2) : ')),0),2)
                    #Met à jour le plateau si le position est libre
                    err = self.updateBoard(y,x,'O')
                self.X.append([inputs,'O'])
                self.Y.append([outputs.tolist(),self.pos.index((y,x))])
                gamer = 'IA'
            else:
                #Tour de l'IA
                inputs = self.boardConv()
                outputs = self.net.feedforward(inputs)
                
                choice = self.outToChoice(outputs)
                print('IA joue : ')
                err = self.updateBoard(choice[0],choice[1],'X')
                if  err :
                    print('IA tente de jouer sur une position occupée !!!')
                self.X.append([inputs,'X'])
                self.Y.append([outputs.tolist(),self.pos.index((choice[0],choice[1]))])
                gamer = 'HU'
            self.printBoard(self.board)
            #Vérification de l'état de la partie
            vict = self.testVictory()
            if vict != '-' :
                if vict == 'O' :
                    print('Vous avez gagné !!!')
                    t = self.learn(self.X,self.Y,100,1.)
                    print("l'IA a appris de nouveaux coups en {} secondes".format(t))
                elif vict == 'X' :
                    print('Vous avez perdu !!!')
                    t = self.learn(self.X,self.Y,100,1.)
                    print("l'IA a appris de nouveaux coups en {} secondes".format(t))
                else :
                    print("C'est un match nul !")
                break

    def learn(self,X,Y,steps,learning_rate):
        for i in range(1,len(Y)+1,2):
            p = Y[-i][0][Y[-i][1]]
            Y[-i][0][Y[-i][1]] = p + (1-p)*(1-i/10)
        for i in range(2,len(Y)+1,2):
            p = Y[-i][0][Y[-i][1]]
            Y[-i][0][Y[-i][1]] = p - p*(1-i/10)
        Y=np.asarray([i[0] for i in Y])
        X=np.asarray([i[0] for i in X])
        t = self.net.train(X, Y, steps=100,batch_size=1000, learning_rate=1.0)
        return t
    
    def printBoard(self,board):
         '''
         cette fonction permet d'afficher le  plateau de jeu dans le terminal
         exemple :
         "[0,0,0]
          [0,0,0]
          [0,0,0]"
         '''
         for i in self.board:
            print(i[0],i[1],i[2],sep='|')



    def updateBoard(self,x,y,sign) :
        '''
        Cette fonction remplace la position symbolisée par x et y par le symbole du joueur
        sur le plateau board, si la position est occupée la variable erreur retourne True,
        le plateau est retourné dans tous les cas (mis a jour si la position était libre)
        '''

        if self.board[x][y] == '-' :
            self.board[x][y] = sign
            
            error = False
        else :
            error = True

        return error


    def boardConv(self) :
        '''
        Cette fonction transforme un plateau de jeu en entrée lisible par une IA
        18 entrée : case 1 -> IA,Humain   case 2 -> IA Humain etc...
        L'entrée lisible par l'IA est retournée par la fonction
        '''

        inputs = [(0,0) if i=='-' else (1,0) if i=='X' else (0,1) for slist in self.board for i in slist]
        inputs = [i for slist in inputs for i in slist]
        return inputs

    def testVictory(self) :
        '''
        Cette fonction verifie un plateau pour voir si l'un des 2 joueurs rempli les conditions de victoire ou
        si le jeu fini en égalité
        si victoire, la fonction retourne le numero du gagnant
        si égalité, la fonction retourne 0
        si la partie continue, la fonction retourne None
        '''
        board = self.board
        for j in range(3):
            column = [board[i][j] for i in range(0,3)]
            print(column)
            if column[1:]== column[:-1]:
                return column[0]
            line = board[j]
            if line[1:]== line[:-1]:
                return line[0]
        diag1 = [board[i][-i-1] for i in range(len(board))]
        if diag1[1:]== diag1[:-1]:
            return diag1[0]
        diag2 = [board[i][i] for i in range(len(board))]
        if diag2[1:]== diag2[:-1]:
            return diag2[0]
        return next((i for slist in board for i in slist if i=='-'), None)
            
       
    def outToChoice(self, outputs) :
        '''
        Cette fonction à pour objectif de convertir les probabilités en sortie de l'IA
        en liste de positions dans l'ordre de "probabilités de victoire" croissant
        '''
        return self.pos[np.argmax(outputs)]  #Retourne la position de la proba max

# générer toutes les grilles possibles :
datain = []
dataout = []
table = { '00':1.,'10':0.,'01':0.,'11':'error'}
for i in range(262143):
    a = bin(i)[2:].zfill(18)
    accepted = True
    tp = []
    for i in range(9):
        tp.append(table[a[2*i]+a[2*i+1]])
        if a[2*i]=='1' and a[2*i+1]=='1':
            accepted = False
    if accepted :
        datain.append([float(i) for i in a])
        dataout.append(tp)



X = np.asarray(datain)
Y = np.asarray(dataout)
Y[Y==1.]=0.5

net = Network(input_dim=18)
net.add_layer(50)
net.add_layer(9)

for i in range(100):
    net.train(X, Y, steps=30,batch_size=1000, learning_rate=1.0)

    print(net.error(X,Y))

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

# sample usage
save_object(net, 'net.pkl')

#verifier ligne
def check(x,y,sign, oppsign, board):
    '''
    possible line
    max number of my sign
    max number of opponent sign
    '''
    cross = []
    cross.append([board[y][i] for i in range(3)])
    cross.append([board[i][x] for i in range(3)])
    if x==y :
        diag1 = [board[x][y] for i in range(3)]
        cross.append([board[i][x] for i in range(3)])
    if x+y==2:
        diag2 = [board[x][-y-1] for i in range(3)]
        cross.append([board[i][x] for i in range(3)])
    nbl=0
    nbme=0
    nbyou=0
    if board[y][x]!='-':
        free = 1
    else :
        free = 0
    for line in cross:
        if oppsign not in line :
            nbl+=1
            nbme = max(line.count(sign),nbme)
        elif sign not in line:
            nbyou = max(line.count(oppsign),nbyou)
    return free, nbl, nbme,nbyou


        
  
