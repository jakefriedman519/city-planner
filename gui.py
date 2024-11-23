import pygame

class GUI:
    GRID_SIZE = 10
    CELL_SIZE = 60
    BUFFER = 100
    LEGEND_WIDTH = 150
    SCREEN_WIDTH = GRID_SIZE * CELL_SIZE + LEGEND_WIDTH
    SCREEN_HEIGHT = GRID_SIZE * CELL_SIZE + BUFFER
    FPS = 60

    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GRAY = (200, 200, 200)

    BUILDING_COLORS = {
        0: (255, 255, 255),  # Empty
        1: (255, 0, 0),      # School
        2: (0, 255, 0),      # Hospital
        3: (0, 0, 255),      # Factory
        4: (255, 255, 0),    # Residential
        5: (0, 255, 255),    # Park
        6: (255, 0, 255),    # Restaurant
    }

    def __init__(self, env):
        pygame.init()
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("City Planning")
        self.clock = pygame.time.Clock()
        self.env = env
        self.current_building = 0

    def draw_grid(self):
        for x in range(0, self.GRID_SIZE * self.CELL_SIZE, self.CELL_SIZE):
            for y in range(self.BUFFER, self.SCREEN_HEIGHT, self.CELL_SIZE):
                rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, self.BLACK, rect, 1)

    def draw_buildings(self, grid):
        for i in range(self.GRID_SIZE):
            for j in range(self.GRID_SIZE):
                rect = pygame.Rect(
                    j * self.CELL_SIZE, i * self.CELL_SIZE + self.BUFFER, self.CELL_SIZE, self.CELL_SIZE
                )
                pygame.draw.rect(self.screen, self.BUILDING_COLORS[grid[i, j]], rect)

    def get_grid_pos(self, mouse_pos):
        return (mouse_pos[1] - self.BUFFER) // self.CELL_SIZE, mouse_pos[0] // self.CELL_SIZE

    def draw_legend(self):
        legend_x = self.GRID_SIZE * self.CELL_SIZE + 10
        legend_y = self.BUFFER
        font = pygame.font.Font(None, 24)

        for i, (building_type, color) in enumerate(self.BUILDING_COLORS.items()):
            if building_type == 0:
                continue

            # Draw color box
            rect = pygame.Rect(legend_x, legend_y + i * 30, 20, 20)
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, self.BLACK, rect, 1)

            # Draw building type text
            building_name = self.env.building_map[building_type - 1]
            text = font.render(building_name, True, self.BLACK)
            self.screen.blit(text, (legend_x + 30, legend_y + i * 30))

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        row, col = self.get_grid_pos(event.pos)
                        if 0 <= row < self.GRID_SIZE and 0 <= col < self.GRID_SIZE:
                            action = (row, col, self.current_building)
                            _, reward, done, _ = self.env.step(action)
                            print(f"Reward: {reward}, Done: {done}")
                    elif event.button == 3:  # Right click
                        self.current_building = (self.current_building + 1) % len(self.env.building_map)
                        print(f"Selected building: {self.env.building_map[self.current_building]}")

            self.screen.fill(self.WHITE)

            font = pygame.font.Font(None, 36)
            budget_text = font.render(f"Budget: ${self.env.remaining_budget}", True, self.BLACK)
            self.screen.blit(budget_text, (10, 10))

            building = self.env.building_map[self.current_building]
            building_text = font.render(
                f"Current: {building}, Price: {self.env.building_costs[building]}",
                True,
                self.BLACK,
            )
            self.screen.blit(building_text, (10, 50))

            self.draw_buildings(self.env.grid)
            self.draw_grid()
            self.draw_legend()

            pygame.display.flip()
            self.clock.tick(self.FPS)

        pygame.quit()

if __name__ == "__main__":
    from city_planner_gym import CityPlanningEnv

    env = CityPlanningEnv()
    game = GUI(env)
    game.run()