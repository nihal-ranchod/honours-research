import pandas as pd
import matplotlib.pyplot as plt
import ast

# Read the results from the file
results = []
with open("evaluation_results.txt", "r") as file:
    current_result = {}
    for line in file:
        line = line.strip()
        if line.endswith(':'):
            if current_result:
                results.append(current_result)
            current_result = {'match': line[:-1]}
        elif line:
            key, value = line.split(": ", 1)
            if key in ['move_times', 'final_elo_bot1', 'final_elo_bot2']:
                current_result[key] = ast.literal_eval(value)
            else:
                current_result[key] = int(value) if value.isdigit() else float(value)
    if current_result:
        results.append(current_result)

# Convert the results to a pandas DataFrame
df = pd.DataFrame(results)
df.set_index('match', inplace=True)

# Calculate additional metrics
df['total_games'] = df['bot1_wins'] + df['bot2_wins'] + df['draws']
df['bot1_win_rate'] = df['bot1_wins'] / df['total_games']
df['bot2_win_rate'] = df['bot2_wins'] / df['total_games']
df['average_game_length'] = df['total_moves'] / df['total_games']
df['average_time_per_game'] = df['total_time'] / df['total_games']

# Calculate average move times for bot1 and bot2
df['average_move_time_bot1'] = df['move_times'].apply(lambda x: sum(x[0]) / len(x[0]) if x[0] else 0)
df['average_move_time_bot2'] = df['move_times'].apply(lambda x: sum(x[1]) / len(x[1]) if x[1] else 0)

# Plot win rates
plt.figure(figsize=(12, 6))
plt.bar(df.index, df['bot1_win_rate'], label='Bot 1 Win Rate', alpha=0.6)
plt.bar(df.index, df['bot2_win_rate'], label='Bot 2 Win Rate', alpha=0.6, bottom=df['bot1_win_rate'])
plt.xlabel('Match')
plt.ylabel('Win Rate')
plt.title('Win Rates of Bots')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot average game length
plt.figure(figsize=(12, 6))
plt.bar(df.index, df['average_game_length'])
plt.xlabel('Match')
plt.ylabel('Average Game Length (moves)')
plt.title('Average Game Length')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot average time per game
plt.figure(figsize=(12, 6))
plt.bar(df.index, df['average_time_per_game'])
plt.xlabel('Match')
plt.ylabel('Average Time per Game (seconds)')
plt.title('Average Time per Game')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot average move times for bot1 and bot2
plt.figure(figsize=(12, 6))
plt.bar(df.index, df['average_move_time_bot1'], label='Bot 1 Average Move Time', alpha=0.6)
plt.bar(df.index, df['average_move_time_bot2'], label='Bot 2 Average Move Time', alpha=0.6, bottom=df['average_move_time_bot1'])
plt.xlabel('Match')
plt.ylabel('Average Move Time (seconds)')
plt.title('Average Move Times of Bots')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot final Elo ratings
plt.figure(figsize=(12, 6))
plt.bar(df.index, df['final_elo_bot1'], label='Bot 1 Final Elo', alpha=0.6)
plt.bar(df.index, df['final_elo_bot2'], label='Bot 2 Final Elo', alpha=0.6, bottom=df['final_elo_bot1'])
plt.xlabel('Match')
plt.ylabel('Final Elo Rating')
plt.title('Final Elo Ratings of Bots')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print summary statistics
print("Summary Statistics:")
print(df[['bot1_win_rate', 'bot2_win_rate', 'average_game_length', 'average_time_per_game', 'average_move_time_bot1', 'average_move_time_bot2', 'final_elo_bot1', 'final_elo_bot2']])
