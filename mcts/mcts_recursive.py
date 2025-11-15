import numpy as np

class Node:
    def __init__(self, state, untried_actions, action=None, parent=None):
        self.state = state
        self.parent = parent
        self.action = action    # action that led to this node
        self.children = []
        self.untried_actions = untried_actions

        # statistics
        self.Q_sa = {}     # average return for (state, action)
        self.N_sa = {}     # visits to (s,a)
        self.N = 0          # visits to this state

        for a in self.untried_actions:
            self.Q_sa[a] = 0.0
            self.N_sa[a] = 0

class MCTS:
    def __init__(self, model, iters=10000, max_depth=10, c=1.4, gamma=1):
        self.max_iterations = iters
        self.max_depth = max_depth
        self.c = c
        self.gamma = gamma
        self.mdp = model

    def get_best_root_action(self, root_state):
        root = Node(root_state, untried_actions=self.mdp.actions(root_state))
        depth = 0

        for _ in range(self.max_iterations):
            self.build_tree(root, depth)
            
        best_action = max(root.Q_sa, key=root.Q_sa.get)
        return best_action

    def selection(self, node):
        best_score = float("-inf")
        best_action = None

        actions = self.mdp.actions(node.state)

        # if we reach here, all available actions have been visited at least once
        # Compute UCB1
        for a in actions: 
            Q_sa = node.Q_sa[a]
            N_sa = node.N_sa[a]

            score = Q_sa + self.c*np.sqrt(np.log(node.N)/N_sa)

            if score > best_score:
                best_score = score
                best_action = a
                    
        return best_action
    
    def expand(self, action, node):
        next_state, _ = self.mdp.step(node.state, action)

        actions = self.mdp.actions(next_state)
        child = Node(next_state, actions, action, node)

        node.children.append(child)

        return child

    def rollout(self, state, cur_node_depth):
        total_return = 0.0
        discount = 1.0
        depth = cur_node_depth

        while depth < self.max_depth:
            actions = self.mdp.actions(state)
            if not actions:
                break  # terminal state

            action = self.mdp.rollout_policy(state)
            next_state, reward = self.mdp.step(state, action)

            total_return += discount * reward
            discount *= self.gamma

            state = next_state
            depth += 1

        return total_return
    
    def backpropagate(self, node, total_return):
        roll_discounted_return = total_return
        current = node

        while current is not None:
            current.N += 1

            if current.parent is not None:
                roll_discounted_return = self.gamma * roll_discounted_return

                action = current.action

                current.parent.N_sa[action] += 1

                Q_old = current.parent.Q_sa[action]
                current.parent.Q_sa[action] = Q_old + (roll_discounted_return - Q_old) / current.parent.N_sa[action]

            current = current.parent

    def build_tree(self, node, depth):
        if depth == self.max_depth:
            value = 0.0
            self.backpropagate(node, value)
            return value

        if node.untried_actions:
            action = node.untried_actions.pop()
            child = self.expand(action, node)

            value = self.rollout(child.state, depth + 1)
            self.backpropagate(child, value)
            return value

        if node.children:
            action = self.selection(node)
            child = next(c for c in node.children if c.action == action)

            return self.build_tree(child, depth + 1)

        value = self.rollout(node.state, depth)
        self.backpropagate(node, value)
        
        return value