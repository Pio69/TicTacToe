import csv
import pickle
from tkinter import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import random

size_of_board = 600
symbol_size = (size_of_board / 3 - size_of_board / 8) / 2
symbol_thickness = 50
symbol_X_color = '#EE4035'
symbol_O_color = '#0492CF'
Green_color = '#7BC043'

# Carregar o modelo treinado
with open('tic_tac_toe_random_forest_grid_search.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

class Tic_Tac_Toe():
    def __init__(self):
        self.window = Tk()
        self.window.title('Tic-Tac-Toe')
        self.canvas = Canvas(self.window, width=size_of_board, height=size_of_board)
        self.canvas.pack()
        self.window.bind('<Button-1>', self.click)

        self.initialize_board()
        self.player_X_turns = True
        self.board_status = np.zeros(shape=(3, 3))

        self.player_X_starts = True
        self.reset_board = False
        self.gameover = False
        self.tie = False
        self.X_wins = False
        self.O_wins = False

        self.X_score = 0
        self.O_score = 0
        self.tie_score = 0

        # Abertura do arquivo CSV para armazenar os dados das jogadas
        self.data_logger = open('tic_tac_toe_data.csv', mode='a', newline='')
        self.csv_writer = csv.writer(self.data_logger)
        self.csv_writer.writerow(['Jogada', 'Tabuleiro', 'Resultado'])

    def mainloop(self):
        self.window.mainloop()

    def initialize_board(self):
        for i in range(2):
            self.canvas.create_line((i + 1) * size_of_board / 3, 0, (i + 1) * size_of_board / 3, size_of_board)
        for i in range(2):
            self.canvas.create_line(0, (i + 1) * size_of_board / 3, size_of_board, (i + 1) * size_of_board / 3)

    def play_again(self):
        # Limpar e redesenhar o tabuleiro
        self.canvas.delete("all")
        self.initialize_board()

        # Redefinir variáveis de controle
        self.board_status = np.zeros(shape=(3, 3))  # Tabuleiro vazio
        self.reset_board = False  # Aguardar a próxima jogada
        self.player_X_starts = not self.player_X_starts  # Alternar quem começa
        self.player_X_turns = self.player_X_starts  # Redefinir o turno
        self.X_wins = False
        self.O_wins = False
        self.tie = False
        self.gameover = False

        # Fechar o arquivo de log anterior e abrir um novo
        self.data_logger.close()  # Fechar o logger atual
        self.data_logger = open('tic_tac_toe_data.csv', mode='a', newline='')  # Reabrir o arquivo de log
        self.csv_writer = csv.writer(self.data_logger)
        self.csv_writer.writerow(['Jogada', 'Tabuleiro', 'Resultado'])

        print("Tabuleiro reiniciado e pronto para uma nova partida.")

    def draw_O(self, logical_position):
        logical_position = np.array(logical_position)
        grid_position = self.convert_logical_to_grid_position(logical_position)
        self.canvas.create_oval(grid_position[0] - symbol_size, grid_position[1] - symbol_size,
                                grid_position[0] + symbol_size, grid_position[1] + symbol_size, width=symbol_thickness,
                                outline=symbol_O_color)

    def draw_X(self, logical_position):
        grid_position = self.convert_logical_to_grid_position(logical_position)
        self.canvas.create_line(grid_position[0] - symbol_size, grid_position[1] - symbol_size,
                                grid_position[0] + symbol_size, grid_position[1] + symbol_size, width=symbol_thickness,
                                fill=symbol_X_color)
        self.canvas.create_line(grid_position[0] - symbol_size, grid_position[1] + symbol_size,
                                grid_position[0] + symbol_size, grid_position[1] - symbol_size, width=symbol_thickness,
                                fill=symbol_X_color)

    def display_gameover(self):
        if self.X_wins:
            self.X_score += 1
            text = 'Winner: Player 1 (X)'
            color = symbol_X_color
        elif self.O_wins:
            self.O_score += 1
            text = 'Winner: Player 2 (O)'
            color = symbol_O_color
        else:
            self.tie_score += 1
            text = 'It\'s a tie'
            color = 'gray'

        self.canvas.delete("all")
        self.canvas.create_text(size_of_board / 2, size_of_board / 3, font="cmr 60 bold", fill=color, text=text)

        score_text = 'Scores \n'
        self.canvas.create_text(size_of_board / 2, 5 * size_of_board / 8, font="cmr 40 bold", fill=Green_color,
                                text=score_text)

        score_text = f'Player 1 (X) : {self.X_score}\nPlayer 2 (O): {self.O_score}\nTie                    : {self.tie_score}'
        self.canvas.create_text(size_of_board / 2, 3 * size_of_board / 4, font="cmr 30 bold", fill=Green_color,
                                text=score_text)
        self.reset_board = True

        self.canvas.create_text(size_of_board / 2, 15 * size_of_board / 16, font="cmr 20 bold", fill="gray",
                                text='Click to play again \n')

    def convert_logical_to_grid_position(self, logical_position):
        logical_position = np.array(logical_position, dtype=int)
        return (size_of_board / 3) * logical_position + size_of_board / 6

    def convert_grid_to_logical_position(self, grid_position):
        grid_position = np.array(grid_position)
        return np.array(grid_position // (size_of_board / 3), dtype=int)

    def is_grid_occupied(self, logical_position):
        return self.board_status[logical_position[0]][logical_position[1]] != 0

    def is_winner(self, player):
        player = -1 if player == 'X' else 1
        for i in range(3):
            if self.board_status[i][0] == self.board_status[i][1] == self.board_status[i][2] == player:
                return True
            if self.board_status[0][i] == self.board_status[1][i] == self.board_status[2][i] == player:
                return True
        if self.board_status[0][0] == self.board_status[1][1] == self.board_status[2][2] == player:
            return True
        if self.board_status[0][2] == self.board_status[1][1] == self.board_status[2][0] == player:
            return True
        return False

    def is_tie(self):
        return np.all(self.board_status != 0)

    def is_gameover(self):
        self.X_wins = self.is_winner('X')
        if not self.X_wins:
            self.O_wins = self.is_winner('O')
        if not self.O_wins:
            self.tie = self.is_tie()
        return self.X_wins or self.O_wins or self.tie

    def predict_O_move(self):
        # Transforma o estado do tabuleiro em uma lista unidimensional
        board_flat = self.board_status.flatten().tolist()
        
        # Verificar o estado do tabuleiro antes da previsão
        #print(f"Tabuleiro atual para a IA (antes da jogada): {board_flat}")
        
        # Fazer a previsão da IA
        predicted_move = model.predict([board_flat])[0]
        
        # Verificar a jogada prevista
        #print(f"Jogada prevista pela IA: {predicted_move}")
        
        # Converte a posição prevista para coordenadas 2D do tabuleiro
        return np.unravel_index(predicted_move, (3, 3))

    def make_ai_move(self):
        if not self.is_gameover():
            # Primeiro, verificar se precisamos bloquear o jogador X
            blocking_move = self.find_blocking_move()
            
            if blocking_move:
                print(f"Bloqueando o jogador X na posição {blocking_move}")
                O_move = blocking_move
            else:
                # Se não houver necessidade de bloquear, prever a jogada da IA
                O_move = self.predict_O_move()

                # Verifica se a posição está ocupada e encontra outra jogada válida
                attempts = 0
                while self.is_grid_occupied(O_move) and attempts < 10:
                    print(f"Posição {O_move} já está ocupada. Prevendo nova jogada.")
                    O_move = self.predict_O_move()
                    attempts += 1
                
                # Se o modelo não conseguir prever uma jogada válida, escolhe uma posição vazia aleatória
                if self.is_grid_occupied(O_move):
                    print("Todas as tentativas falharam. Selecionando uma jogada aleatória.")
                    available_positions = [(i, j) for i in range(3) for j in range(3) if self.board_status[i][j] == 0]
                    O_move = random.choice(available_positions)

            # Fazer a jogada da IA (O) na posição disponível
            self.draw_O(O_move)
            self.board_status[O_move[0]][O_move[1]] = 1
            self.player_X_turns = not self.player_X_turns
            self.log_data('O')  # Log da jogada de O

            if self.is_gameover():
                self.display_gameover()


    def log_data(self, player):
        board_flat = self.board_status.flatten().tolist()
        result = 1 if self.X_wins else (-1 if self.O_wins else 0.5)
        self.csv_writer.writerow([player, board_flat, result])

    def click(self, event):
        grid_position = [event.x, event.y]
        logical_position = self.convert_grid_to_logical_position(grid_position)

        if not self.reset_board:
            if self.player_X_turns:
                if not self.is_grid_occupied(logical_position):
                    self.draw_X(logical_position)
                    self.board_status[logical_position[0]][logical_position[1]] = -1
                    self.player_X_turns = not self.player_X_turns
                    self.log_data('X')  # Log da jogada de X

                    if self.is_gameover():
                        self.display_gameover()
                    else:
                        # IA faz a jogada para O
                        self.make_ai_move()

        else:
            self.canvas.delete("all")
            self.play_again()
            self.reset_board = False

    def __del__(self):
        self.data_logger.close()

    def find_blocking_move(self):
        # Verificar todas as combinações vencedoras
        winning_combinations = [
            [(0, 0), (0, 1), (0, 2)],  # Primeira linha
            [(1, 0), (1, 1), (1, 2)],  # Segunda linha
            [(2, 0), (2, 1), (2, 2)],  # Terceira linha
            [(0, 0), (1, 0), (2, 0)],  # Primeira coluna
            [(0, 1), (1, 1), (2, 1)],  # Segunda coluna
            [(0, 2), (1, 2), (2, 2)],  # Terceira coluna
            [(0, 0), (1, 1), (2, 2)],  # Diagonal principal
            [(0, 2), (1, 1), (2, 0)],  # Diagonal secundária
        ]
        
        # Verificar se o jogador X (-1) está prestes a vencer e encontrar a posição que a IA pode bloquear
        for combo in winning_combinations:
            x_count = sum([1 for pos in combo if self.board_status[pos[0]][pos[1]] == -1])
            empty_count = sum([1 for pos in combo if self.board_status[pos[0]][pos[1]] == 0])
            
            # Se o jogador X tiver 2 marcas e houver 1 espaço vazio, precisamos bloquear
            if x_count == 2 and empty_count == 1:
                for pos in combo:
                    if self.board_status[pos[0]][pos[1]] == 0:
                        return pos  # Retornar a posição vazia para bloquear
        
        return None  # Se não houver necessidade de bloqueio


# Instanciar o jogo e iniciar o mainloop
game_instance = Tic_Tac_Toe()
game_instance.mainloop()
