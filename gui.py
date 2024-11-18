import pygame
from city_planner_gym import CityPlanningEnv

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
    1: (255, 0, 0),  # School
    2: (0, 255, 0),  # Hospital
    3: (0, 0, 255),  # Factory
    4: (255, 255, 0),  # Residential
    5: (0, 255, 255),  # Park
    6: (255, 0, 255),  # Restaurant
}

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("City Planning")
clock = pygame.time.Clock()

env = CityPlanningEnv()


def draw_grid():
    for x in range(0, GRID_SIZE * CELL_SIZE, CELL_SIZE):
        for y in range(BUFFER, SCREEN_HEIGHT, CELL_SIZE):
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BLACK, rect, 1)


def draw_buildings(grid):
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            rect = pygame.Rect(
                j * CELL_SIZE, i * CELL_SIZE + BUFFER, CELL_SIZE, CELL_SIZE
            )
            pygame.draw.rect(screen, BUILDING_COLORS[grid[i, j]], rect)


def get_grid_pos(mouse_pos):
    return (mouse_pos[1] - BUFFER) // CELL_SIZE, mouse_pos[0] // CELL_SIZE


def draw_legend():
    legend_x = GRID_SIZE * CELL_SIZE + 10
    legend_y = BUFFER
    font = pygame.font.Font(None, 24)

    for i, (building_type, color) in enumerate(BUILDING_COLORS.items()):
        if building_type == 0:
            continue

        # Draw color box
        rect = pygame.Rect(legend_x, legend_y + i * 30, 20, 20)
        pygame.draw.rect(screen, color, rect)
        pygame.draw.rect(screen, BLACK, rect, 1)

        # Draw building type text
        building_name = env.building_map[building_type - 1]
        text = font.render(building_name, True, BLACK)
        screen.blit(text, (legend_x + 30, legend_y + i * 30))


def main():
    current_building = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    row, col = get_grid_pos(event.pos)
                    if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
                        action = (row, col, current_building)
                        _, reward, done, _ = env.step(action)
                        print(f"Reward: {reward}, Done: {done}")
                elif event.button == 3:  # Right click
                    current_building = (current_building + 1) % len(env.building_map)
                    print(f"Selected building: {env.building_map[current_building]}")

        screen.fill(WHITE)

        font = pygame.font.Font(None, 36)
        budget_text = font.render(f"Budget: ${env.remaining_budget}", True, BLACK)
        screen.blit(budget_text, (10, 10))

        building = env.building_map[current_building]
        building_text = font.render(
            f"Current: {building}, Price: {env.building_costs[building]}",
            True,
            BLACK,
        )
        screen.blit(building_text, (10, 50))

        draw_buildings(env.grid)
        draw_grid()
        draw_legend()

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()