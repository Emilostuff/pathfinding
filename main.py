import pygame
from perlin_noise import PerlinNoise


class State:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Game:
    INSET = 50
    THRESHOLD = 0.1

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[True for _ in range(width)] for _ in range(height)]
        self.generate_islands()
        self.start = self.starting_state()
        self.goal = self.goal_state()

    def generate_islands(self):
        noise = PerlinNoise(octaves=9, seed=8)
        xpix, ypix = self.width, self.height

        pic = [[noise([i / xpix, j / ypix]) for j in range(xpix)] for i in range(ypix)]
        for y in range(self.height):
            for x in range(self.width):
                if pic[y][x] > Game.THRESHOLD:
                    self.grid[y][x] = False

    def starting_state(self):
        # Find a valid starting state in top left corner
        for y in range(Game.INSET, self.height):
            for x in range(Game.INSET, self.width):
                if self.grid[y][x]:
                    return State(x, y)

    def goal_state(self):
        # Find a valid goal state in bottom right corner
        for y in range(self.height - Game.INSET - 1, 0, -1):
            for x in range(self.width - Game.INSET - 1, 0, -1):
                if self.grid[y][x]:
                    return State(x, y)

    def valid_moves(self, state):
        # Player can move orhoganally and diagonally
        actions = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if 0 <= state.x + dx < self.width and 0 <= state.y + dy < self.height:
                    if self.grid[state.y + dy][state.x + dx]:
                        actions.append(State(state.x + dx, state.y + dy))
        return actions

    def is_goal(self, state):
        return state.x == self.goal.x and state.y == self.goal.y


class Renderer:
    LIGHT = (227, 211, 191)  # Land
    DARK = (56, 49, 59)  # Mountain
    START = (214, 36, 99)  # Red
    GOAL = (55, 204, 65)  # Green

    def __init__(self, game):
        self.game = game

        pygame.init()
        self.window = pygame.display.set_mode((game.width, game.height))
        pygame.display.set_caption("Pathfinding")

    def draw(self):
        for y in range(self.game.height):
            for x in range(self.game.width):
                if self.game.grid[y][x]:
                    self.window.set_at((x, y), Renderer.DARK)
                else:
                    self.window.set_at((x, y), Renderer.LIGHT)

        # Draw start and goal
        pygame.draw.circle(
            self.window, Renderer.START, (self.game.start.x, self.game.start.y), 5
        )
        pygame.draw.circle(
            self.window, Renderer.GOAL, (self.game.goal.x, self.game.goal.y), 5
        )
        pygame.display.flip()


if __name__ == "__main__":
    game = Game(500, 500)
    renderer = Renderer(game)
    renderer.draw()

    # Main loop
    running = True
    while running:
        pygame.display.flip()  # Update display

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                # Quit pygame
                pygame.quit()
