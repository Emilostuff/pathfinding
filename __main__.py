import pygame
from perlin_noise import PerlinNoise
from algoaid import MinHeap
from dataclasses import dataclass
import random
import math


@dataclass(frozen=True)
class State:
    x: int
    y: int


class Game:
    INSET = 10
    THRESHOLD = 0.08
    SEED = random.randint(0, 1000)
    SQRT2 = 2**0.5

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[True for _ in range(width)] for _ in range(height)]
        self.generate_islands()
        self.start = self.starting_state()
        self.goal = self.goal_state()

    def generate_islands(self):
        noise = PerlinNoise(octaves=8, seed=Game.SEED)
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

    def is_valid(self, state):
        return (
            0 <= state.x < self.width
            and 0 <= state.y < self.height
            and self.grid[state.y][state.x]
        )

    def valid_moves(self, state):
        # Player can move orhoganally (cost 1) and diagonally (cost sqrt(2))
        actions = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                new = State(state.x + dx, state.y + dy)
                if self.is_valid(new):
                    if dx == 0 or dy == 0:
                        actions.append((new, 1))
                    else:
                        actions.append((new, Game.SQRT2))

        return actions

    def is_goal(self, state):
        return state.x == self.goal.x and state.y == self.goal.y

    def distance_to_goal(self, state):
        # straight line distance
        return math.hypot(state.x - self.goal.x, state.y - self.goal.y)


class Renderer:
    LIGHT = (227, 211, 191)  # Land
    DARK = (148, 133, 115)  # Mountain
    START = (214, 36, 99)  # Red
    GOAL = (9, 150, 18)  # Green
    PATH = (44, 168, 52)  # Path Green
    EXPLORED = (227, 202, 191)
    FRONTIER = (214, 159, 167)
    TILE_SIZE = 8

    def __init__(self, game):
        self.game = game

        self.window = pygame.display.set_mode(
            (game.width * Renderer.TILE_SIZE + 400, game.height * Renderer.TILE_SIZE)
        )
        pygame.display.set_caption("Pathfinding")

    def draw_pixel(self, x, y, color):
        pygame.draw.rect(
            self.window,
            color,
            (
                x * Renderer.TILE_SIZE,
                y * Renderer.TILE_SIZE,
                Renderer.TILE_SIZE,
                Renderer.TILE_SIZE,
            ),
        )

    def draw_circle(self, x, y, color):
        pygame.draw.circle(
            self.window,
            color,
            (
                x * Renderer.TILE_SIZE + Renderer.TILE_SIZE / 2,
                y * Renderer.TILE_SIZE + Renderer.TILE_SIZE / 2,
            ),
            Renderer.TILE_SIZE,
        )

    def draw(self, path=None, astar=None):
        # Draw map
        for y in range(self.game.height):
            for x in range(self.game.width):
                if self.game.grid[y][x]:
                    self.draw_pixel(x, y, Renderer.LIGHT)
                else:
                    self.draw_pixel(x, y, Renderer.DARK)

        if astar:
            # Draw explored
            for state in astar.explored:
                self.draw_pixel(state.x, state.y, Renderer.EXPLORED)
            # Draw frontier
            for state in astar.in_frontier:
                self.draw_pixel(state.x, state.y, Renderer.FRONTIER)

        # Draw path
        if path:
            for state in path:
                self.draw_pixel(state.x, state.y, Renderer.PATH)

        # Draw start and goal
        self.draw_circle(self.game.start.x, self.game.start.y, Renderer.START)
        self.draw_circle(self.game.goal.x, self.game.goal.y, Renderer.GOAL)

        # test
        font = pygame.freetype.Font("assets/menlo.ttc", 24)
        text = font.render("A* WEIGHT", (255, 255, 255), (0, 0, 0))
        self.window.blit(text[0], dest=(800, 10))

        pygame.display.flip()


class Node:
    def __init__(self, state, parent, path_cost):
        self.state = state
        self.parent = parent
        self.path_cost = path_cost


class AStar:
    WEIGHT = 1.2

    def __init__(self, game):
        self.game = game
        self.frontier = MinHeap()
        self.in_frontier = dict()
        self.explored = set()

    def heuristic(self, node):
        return self.game.distance_to_goal(node.state) * AStar.WEIGHT

    def add_to_frontier(self, node):
        # Check if the node is already explored
        if node.state in self.explored:
            return

        # Compute the cost of the node
        cost = node.path_cost + self.heuristic(node)

        # Check if the node is already in the frontier
        if node.state in self.in_frontier:
            current_node = self.in_frontier[node.state]
            current_cost = current_node.path_cost + self.heuristic(current_node)

            # If the new node is better, update the frontier and the node
            if cost < current_cost:
                self.frontier.decrease_key(current_node, cost)
                self.in_frontier[node.state].path_cost = node.path_cost
                self.in_frontier[node.state].parent = node.parent
        else:
            self.frontier.insert(node, cost)
            self.in_frontier[node.state] = node

    def pop_from_frontier(self):
        node = self.frontier.extract_min()
        del self.in_frontier[node.state]
        return node

    def search(self):
        # Add start node to the frontier
        start = Node(self.game.start, None, 0)
        self.add_to_frontier(start)

        while not self.frontier.empty():
            # Get the node with the lowest cost
            current = self.pop_from_frontier()

            # Check if the node is the goal
            if self.game.is_goal(current.state):
                return self.solution(current)

            # Add the node to the explored set
            self.explored.add(current.state)

            # Expand the node
            for state, cost in self.game.valid_moves(current.state):
                child = Node(state, current, current.path_cost + cost)
                self.add_to_frontier(child)

        return None

    def solution(self, node):
        path = []
        while node:
            path.append(node.state)
            node = node.parent
        return path[::-1]


if __name__ == "__main__":
    pygame.init()
    pygame.font.init()
    clock = pygame.time.Clock()

    while True:
        game = Game(100, 100)
        astar = AStar(game)
        if astar.search():
            break

    renderer = Renderer(game)

    # Main loop
    running = True
    while running:
        clock.tick(30)
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Handle movement
        keys = pygame.key.get_pressed()
        x, y = game.start.x, game.start.y
        if keys[pygame.K_LEFT] and game.is_valid(State(x - 1, y)):
            x -= 1
        if keys[pygame.K_RIGHT] and game.is_valid(State(x + 1, y)):
            x += 1
        if keys[pygame.K_UP] and game.is_valid(State(x, y - 1)):
            y -= 1
        if keys[pygame.K_DOWN] and game.is_valid(State(x, y + 1)):
            y += 1

        game.start = State(x, y)

        astar = AStar(game)
        path = astar.search()
        renderer.draw(path=path, astar=astar)

    pygame.quit()
