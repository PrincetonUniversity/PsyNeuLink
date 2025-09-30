import copy
from dataclasses import dataclass
from collections import namedtuple, defaultdict
from msdm.core.mdp import TabularMarkovDecisionProcess
from msdm.core.pomdp import TabularPOMDP
from msdm.core.distributions import DictDistribution

# Door states: O - open, D - Door closed, R - red locked, G - green locked, etc

State = namedtuple("State", "x y door_states key_states holding_key key_color")
Action = namedtuple("Action", "dx dy open pickup")
Observation = namedtuple("Observation", "x y obs_grid door_states key_states holding_key key_color")


class KeysAndDoors(TabularPOMDP):
    def __init__(
            self,
            coherence=.95,
            discount_rate=.95,
            step_cost=-1,
            target_reward=100,
            grid=None
    ):
        """
        Parameters
        ---------
        :coherence:
        :discount_rate:
        :step_cost:
        :reward:
        :grid:            A multiline string.
                          `s` is the initial state,
                          `#` are walls,
                          't' is the target
                          'd' are closed doors
                          'O' are open doors
                          'R', 'G', 'B'... etc are locked doors
                          'r', 'g', 'b'... etc are keys
        """
        if grid is None:
            grid = \
                """
                t....
                ##R##
                .....
                ##s.r
                """
        grid = [list(r.strip()) for r in grid.split('\n') if len(r.strip()) > 0]
        self.grid = grid
        self.loc_features = {}
        self.features_loc = defaultdict(list)

        self.height = len(self.grid)
        self.width = len(self.grid[0])

        self.door_loc = []
        self.key_loc = []
        self.key_colors = []

        # Initialize all grid positions
        for y, row in enumerate(grid):
            for x, f in enumerate(row):
                self.loc_features[(x, y)] = f
                self.features_loc[f].append((x, y))

                if f in 'RGB':
                    self.door_loc.append((x, y))
                elif f in 'rgb':
                    self.key_loc.append((x, y))
                    self.key_colors.append(f)
                elif (f == 'D'):
                    self.door_loc.append((x, y))

        self.coherence = coherence
        self.discount_rate = discount_rate
        self.step_cost = step_cost
        self.target_reward = target_reward

    def initial_state_dist(self):
        x, y = self.features_loc['s'][0]
        temp_stat = list()
        # Change initial status for each closed and locked door
        for door, (doorx, doory) in enumerate(self.door_loc):
            if self.loc_features[doorx, doory] == 'D':
                temp_stat.append('D')
            if self.loc_features[doorx, doory] in 'RGB':
                temp_stat.append(self.loc_features[doorx, doory])
        initial_door_stat = tuple(temp_stat)

        # Initializes all keys as present
        key_states = [True] * len(self.key_loc)

        return DictDistribution({
            State(x=x, y=y, door_states=initial_door_stat, holding_key=False, key_states=tuple(key_states),
                  key_color=None): 1.0,
        })

    # Current Actions: x, y, open door, Pick up key
    def actions(self, s):
        return (
            Action(0, -1, False, False),
            Action(0, 1, False, False),
            Action(-1, 0, False, False),
            Action(1, 0, False, False),
            Action(0, 0, True, False),
            Action(0, 0, False, True),
        )

    def is_absorbing(self, s):
        loc = (s.x, s.y)
        return self.loc_features.get(loc) == 't'

    def next_state_dist(self, s, a):
        x, y = s.x, s.y
        nx, ny = (s.x + a.dx, s.y + a.dy)
        door_states = list(s.door_states)
        key_states = list(s.key_states)
        adjacent = []
        key_state = s.holding_key
        held_color = s.key_color

        # Don't consider states outside of the grid
        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            return DictDistribution({State(x=x, y=y, door_states=tuple(door_states), key_states=tuple(key_states),
                                           holding_key=key_state, key_color=held_color): 1.0})

        # Check if agent is on an edge
        if (x - 1 >= 0):
            adjacent.append((x - 1, y))
        if (x + 1 < self.width):
            adjacent.append((x + 1, y))
        if (y - 1 >= 0):
            adjacent.append((x, y - 1))
        if (y + 1 < self.height):
            adjacent.append((x, y + 1))

        # Pick Up Key
        if a.pickup and not key_state:
            if (x, y) in self.key_loc:
                key_index = self.key_loc.index((x, y))
                if key_states[key_index]:
                    key_states[key_index] = False
                    key_state = True
                    held_color = self.key_colors[key_index]

        # Open Door
        if a.open:
            for adj in adjacent:
                if self.loc_features.get(adj) == 'D':
                    door_index = self.door_loc.index(adj)
                    door_states[door_index] = 'O'
                # adjacent to a door that is not D
                elif adj in self.door_loc:
                    door = self.door_loc.index(adj)
                    if key_state and (held_color == (door_states[door]).lower()):
                        door_states[door] = 'O'
                        key_state = False
                        held_color = None

        # Handles movement for blocked spaces
        if self.loc_features.get((nx, ny), '#') == '#':
            nx, ny = (s.x, s.y)
        if (nx, ny) in self.door_loc:
            if not (door_states[self.door_loc.index((nx, ny))] == 'O'):
                nx, ny = x, y

        return DictDistribution({
            State(x=nx, y=ny, door_states=tuple(door_states), key_states=tuple(key_states), holding_key=key_state,
                  key_color=held_color): 1.0
        })

    def reward(self, s, a, ns):
        r = 0
        r += self.step_cost
        if self.loc_features[(ns.x, ns.y)] == 't':
            r += self.target_reward
        return r

    def observation_dist(self, a, ns):
        obs_grid = set()
        radius = 3

        # Bresenham's line alg. (double check if implemented correctly)
        def line_of_sight(x0, y0, xf, yf):
            dx = abs(xf - x0)
            dy = abs(yf - y0)

            # Determining Direction
            if x0 < xf:
                sx = 1
            else:
                sx = -1
            if y0 < yf:
                sy = 1
            else:
                sy = -1
            decisionParam = dx - dy

            x = x0
            y = y0

            while True:
                # Stop if we hit a wall or our target
                if self.loc_features.get((x, y), '#') == '#' and (x, y) != (xf, yf):
                    return False
                if x == xf and y == yf:
                    return True

                # Determine whether we need to change the y param
                decisionParam2 = 2 * decisionParam
                if decisionParam2 > -dy:
                    decisionParam -= dy
                    x += sx
                if decisionParam2 < dx:
                    decisionParam += dx
                    y += sy

        # Determine which parts of the grid are currently visible
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                cx = ns.x + dx
                cy = ns.y + dy

                if (cx < 0 or cx >= self.width):
                    continue
                if (cy < 0 or cy >= self.height):
                    continue

                if line_of_sight(ns.x, ns.y, cx, cy):
                    obs_grid.add((cx, cy))

        obs_door = []
        # Keep track of which doors are visible
        for door, (doorx, doory) in enumerate(self.door_loc):
            if (doorx, doory) in obs_grid:
                obs_door.append(ns.door_states[door])
            else:
                obs_door.append(None)

        return DictDistribution({
            Observation(x=ns.x, y=ns.y, obs_grid=tuple(obs_grid), door_states=tuple(obs_door), key_states=ns.key_states,
                        holding_key=ns.holding_key, key_color=ns.key_color): 1.0
        })

    def state_string(self, s):
        grid = copy.deepcopy(self.grid)

        for door, (door_x, door_y) in enumerate(self.door_loc):
            if s.door_states[door] == 'O':
                grid[door_y][door_x] = 'O'

        for key_index, (keyx, keyy) in enumerate(self.key_loc):
            if not s.key_states[key_index]:
                grid[keyy][keyx] = '.'
            if s.key_states[key_index] and (grid[keyy][keyx] != self.key_colors[key_index]):
                grid[keyy][keyx] = self.key_colors[key_index]

        for y, row in enumerate(grid):
            for x, f in enumerate(row):
                if (x, y) == (s.x, s.y):
                    grid[y][x] = '@'
        return '\n'.join([''.join(r) for r in grid])
