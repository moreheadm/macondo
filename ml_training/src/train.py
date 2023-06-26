
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import sys
import csv
import tqdm

from dqn_trainer import *
from network import *
from replay_buffer import *

DEVICE = 'cuda'

ENGLISH_LETTER_DIST = {
    '?': 2,
    'E': 12,
    'A': 9,
    'I': 9,
    'O': 8,
    'N': 6,
    'R': 6,
    'T': 6,
    'L': 4,
    'S': 4,
    'U': 4,
    'D': 4,
    'G': 3,
    'B': 2,
    'C': 2,
    'M': 2,
    'P': 2,
    'F': 2,
    'H': 2,
    'V': 2,
    'W': 2,
    'Y': 2,
    'K': 1,
    'J': 1,
    'X': 1,
    'Q': 1,
    'Z': 1
}
csv.field_size_limit(sys.maxsize)

class Board:
    def __init__(self, board=None, spread_value=1.):
        self.bag_offset = 15 * 15 * 27
        self.leave_offset = self.bag_offset + 27
        self.spread_offset = self.leave_offset + 27
        if board is None:
            self.board = torch.zeros((self.spread_offset + 2,))
        else:
            self.board = board

        if spread_value is not None: self.board[self.spread_offset + 1] = spread_value


    def set_letter_dist(self, dist):
        for ch in dist:
            if ch == '?': self.board[self.bag_offset + 26] = dist[ch]
            else: self.board[self.bag_offset + ord(ch) - ord('A')] = dist[ch]

    def apply_play(self, play, spread):
        self.board[self.spread_offset] = spread
        play = play.strip()

        # TODO: improve this
        if play.startswith('(exch'):
            for ch in play[6:play.find(')')]:
                if ch == '?': self.board[self.bag_offset + 26] += 1.
                else: self.board[self.bag_offset + ord(ch) - ord('A')] += 1
            return
        elif play.startswith('(Pass'):
            return

        space, word = play.strip().split()
        if space[0].isnumeric():
            # horizontal
            row_idx = int(space[:-1]) - 1
            col_idx = ord(space[1]) - ord('A')
            mult = 1

        else:
            row_idx = int(space[1:]) - 1
            col_idx = ord(space[1]) - ord('A')
            mult = 15
        
        idx = row_idx * 15 + col_idx
        for i, ch in enumerate(word):
            if ch == '.': continue

            if ch.islower():
                ch = ch.upper()
                ch_idx = ord(ch) - ord('A')
                self.board[225 * 26 + idx + i] = 1.
                self.board[self.bag_offset + 26] -= 1.
            else:
                ch_idx = ord(ch) - ord('A')
                self.board[self.bag_offset + ch_idx] -= 1.
            
            self.board[225 * ch_idx  + idx + mult * i] = 1.


    def get_board(self):
        return self.board.to_sparse()

    def incorporate_leave(self, leave):
        leave = leave.strip()
        for ch in leave:
            ch_idx = 26 if ch == '?' else ord(ch) - ord('A')
            self.board[self.leave_offset + ch_idx] += 1.
            self.board[self.bag_offset + ch_idx] -= 1.
        return self

    def clone(self):
        return Board(self.board.clone(), spread_value=None)

def sign(x):
    if x > 0:
        return 1.
    elif x < 0:
        return -1.
    else:
        return 0.

class AutoplayConverter:
    def __init__(self, autoplay_filename):
        self.autoplay_filename = autoplay_filename
        self.games = {}
        self.experiences = []

    def convert(self):
        with open(self.autoplay_filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for move in tqdm.tqdm(reader):
                self.games.setdefault(move['gameID'], []).append(move)
                if move['leave'].strip() == '':
                    try:
                        self.convert_game(self.games[move['gameID']])
                    except:
                        print('Failed finished game:', self.games[move['gameID']], file=sys.stderr)
                        
                    self.games.pop(move['gameID'])

            for game in self.games:
                try:
                    self.convert_game(game)
                except:
                    print('Failed unfinished game:', game, file=sys.stderr)

    def convert_game(self, game):
        board = Board()
        board.set_letter_dist(ENGLISH_LETTER_DIST)
        states = []
        afterstates = []
        rewards = []
        dones = []
        next_states = []
        next_actions = []

        for i, move in enumerate(game):
            states.append(board.get_board())
            board.apply_play(move['play'], float(move['totalscore']) - float(move['oppscore']))
            afterstates.append(board.clone().incorporate_leave(move['leave']).get_board())

        num_moves = len(afterstates)
        for i in range(num_moves):
            if i + 2 >= num_moves:
                dones.append(1.)

                if i + 1 == num_moves:
                    rewards.append(sign(float(game[i]['totalscore']) - float(game[i]['oppscore'])) * 5000.)
                else:
                    rewards.append(sign(float(game[i + 1]['totalscore']) - float(game[i + 1]['oppscore']))
                                   * -5000.- float(game[i + 1]['score']))

                next_states.append(Board(spread_value=None).get_board())
                next_actions.append(torch.zeros((0, next_states[-1].size()[0])).to_sparse().coalesce())
            else:
                next_states.append(states[i + 2])
                next_actions.append(self.convert_next_actions(game[i + 2]['moves'], game[i + 2]['rack']))
                rewards.append(-float(game[i + 1]['score']))
                dones.append(0.)

        for ex in zip(afterstates, rewards, next_states, next_actions, dones):
            self.experiences.append(Experience(*ex))

    def get_experiences(self):
        return self.experiences


    def convert_next_actions(self, moves, rack):
        moves = [s.strip() for s in moves.split(';')]
        actions = []

        for move_str in moves:
            board = Board(spread_value=None)
            if move_str.startswith('(exch'):
                tiles_played = move_str[6:move_str.find(')')]
            elif move_str.startswith('(Pass)'):
                tiles_played = ''
            else:
                move = tuple(move_str.split())
                board.apply_play(move[0] + ' ' + move[1], float(move[2]))
                tiles_played = move[1]

            leave = [ch for ch in rack]

            for ch in tiles_played:
                if ch.islower(): leave.remove('?')
                elif ch == '.': continue
                else: leave.remove(ch)

            board.incorporate_leave(''.join(leave))
            actions.append(board.get_board().unsqueeze(0))

        return torch.cat(actions).coalesce()

class QTrainer:
    def __init__(self, checkpoint_file=None, buffer_filename=None, capacity=None):
        if checkpoint_file:
            if os.path.exists(checkpoint_file):
                self.load_state(checkpoint_file, buffer_filename, capacity)
            else:
                print(f'Input file {checkpoint_file} does not exist', file=sys.stderr)
                exit(1)
        else:
            if capacity is None:
                capacity = 10000

            self.policy_net = NNUE()
            self.replay_buffer = ReplayBuffer(capacity)
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-5, weight_decay=1e-5)
        self.trainer = None

    def initialize(self):
        self.policy_net.initialize_parameters()
        
    def load_state(self, filename,  buffer_filename, capacity=None):
        checkpoint = torch.load(filename)

        if buffer_filename is not None:
            replay_buffer_checkpoint = torch.load(buffer_filename)

            if capacity is None:
                capacity = len(replay_buffer_checkpoint['replay_buffer'])
            self.replay_buffer = ReplayBuffer(capacity)
            self.replay_buffer.buffer = replay_buffer_checkpoint['replay_buffer']

        self.policy_net = NNUE().to(DEVICE)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])

        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def save_state(self, filename, buffer_filename):
        saved_object = {
            'policy_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        if self.trainer is not None: saved_object['training_stats'] = self.trainer.training_stats
        torch.save(saved_object, filename)

        if buffer_filename is not None:
            torch.save({
                'replay_buffer': self.replay_buffer.buffer,
            }, buffer_filename)

    def add_data_to_replay_buffer(self, capacity):
        # Add your logic here for adding new data to the replay buffer
        pass

    def convert_autoplay_file(self, autoplay_filename):
        converter = AutoplayConverter(autoplay_filename)
        converter.convert()
        print(f'Converted {len(converter.get_experiences())} experiences')
        for experience in converter.get_experiences():
            self.replay_buffer.push(experience)
            print('Size: ', experience.get_size())

    def train(self, steps):
        target_net = NNUE().to(DEVICE)
        target_net.load_state_dict(self.policy_net.state_dict())
        self.trainer = DQNTrainer(self.policy_net, target_net, self.replay_buffer, self.optimizer)
        self.trainer.train(steps)

    def add_data_to_replay_buffer(self, capacity):
        # Add your logic here for adding new data to the replay buffer
        pass

    def convert_autoplay_file(self, autoplay_filename):
        converter = AutoplayConverter(autoplay_filename)
        converter.convert()
        print(f'Converted {len(converter.get_experiences())} experiences')
        for experience in converter.get_experiences():
            self.replay_buffer.push(experience)
            print('Size: ', experience.get_size())


def main():
    parser = argparse.ArgumentParser(description='Train Q-learning model.')
    parser.add_argument('mode', choices=['initialize', 'add_data', 'train', 'convert'])
    parser.add_argument('-i', '--input-filename', help='Input checkpoint file')
    parser.add_argument('-b', '--buffer-filename', help='Either autoplay.txt or buffer file')
    parser.add_argument('-o', '--output-filename', required=True, help='Output checkpoint file')
    parser.add_argument('-s', '--steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('-c', '--capacity', type=int, help='Replay buffer capacity')
    args = parser.parse_args()

    qt = QTrainer(args.input_filename, args.buffer_filename, args.capacity)

    if args.mode == 'initialize':
        qt.initialize()
    elif args.mode == 'add_data':
        qt.add_data_to_replay_buffer(args.capacity)
    elif args.mode == 'convert':
        qt.convert_autoplay_file(args.buffer_filename)
    elif args.mode == 'train':
        qt.train(args.steps)

    qt.save_state(args.output_filename, None if args.mode != convert else args.buffer_filename)


if __name__ == '__main__':
    main()

