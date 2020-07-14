import numpy as np
import pygame
from PIL import Image


def boot_display():
    # Pygame testing
    scale = 14
    pygame.init()
    window = pygame.display.set_mode((scale * 28, scale * 28 + 40))
    # establishing the way the window will look on launch
    # creating the grid in accordance with the scale
    for x in range(28):
        pygame.draw.rect(window, (55, 55, 55), (x * scale, 0, 1, scale * 28))
    for y in range(29):
        pygame.draw.rect(window, (55, 55, 55), (0, y * scale, scale * 28, 1))

    # creating a 'CLEAR' button, later used to wipe the screen
    font = pygame.font.SysFont('timesnewroman', 20)
    clear_button = font.render('CLEAR', True, (0, 0, 0), (200, 200, 200))
    clear_rec = clear_button.get_rect()
    clear_rec.center = (int(clear_rec[2] / 2) + 10, 28 * scale + 20)
    window.blit(clear_button, clear_rec)

    # creating a 'GUESS NUMBER' button, used to run the image through the NN
    guess_button = font.render('GUESS', True, (0, 0, 0), (200, 200, 200))
    guess_rec = clear_button.get_rect()
    guess_rec.center = (120, 28 * scale + 20)
    window.blit(guess_button, guess_rec)

    # creating a display for the top 3 guesses of what the number is
    #best guess
    guess_text = []
    guess1_button = font.render('3, 0.795', True, (0, 0, 0), (200, 200, 200))
    guess1_rec = clear_button.get_rect()
    guess1_rec.center = (200, 28 * scale + 20)
    guess_text.append([guess1_button, guess1_rec])

    # second best guess
    guess2_button = font.render('3, 0.795', True, (0, 0, 0), (200, 200, 200))
    guess2_rec = clear_button.get_rect()
    guess2_rec.center = (270, 28 * scale + 20)
    guess_text.append([guess2_button, guess2_rec])

    # third best guess
    guess3_button = font.render('3, 0.795', True, (0, 0, 0), (200, 200, 200))
    guess3_rec = clear_button.get_rect()
    guess3_rec.center = (340, 28 * scale + 20)
    guess_text.append([guess3_button, guess3_rec])

    running = True
    drawing = False
    while running:

        pygame.time.delay(15)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
                # if the 'CLEAR' button is clicked on, clear the screen and re-draw the grid and button
                if pygame.mouse.get_pos()[1] > scale * 28 and 10 < pygame.mouse.get_pos()[0] < 72:
                    window.fill((0, 0, 0))
                    for x in range(28):
                        pygame.draw.rect(window, (55, 55, 55), (x * scale, 0, 1, scale * 28))
                    for y in range(29):
                        pygame.draw.rect(window, (55, 55, 55), (0, y * scale, scale * 28, 1))
                    window.blit(clear_button, clear_rec)
                    window.blit(guess_button, guess_rec)
                # if the "GUESS" button is pressed, save the image, convert it to a 28 x 28 image
                # and then run it through the neural network to predict what the number is
                elif pygame.mouse.get_pos()[1] > scale * 28 and 88 < pygame.mouse.get_pos()[0] < 150:
                    pygame.image.save(window, "number_guess.jpeg")
                    num_guess = Image.open("number_guess.jpeg", 'r')
                    raw_img = np.array(num_guess)
                    fixed_img = np.zeros(784)
                    i = 0
                    for y in range(int(scale / 2), len(raw_img) - 40, scale):
                        for x in range(int(scale / 2), len(raw_img[0]), scale):
                            fixed_img[i] += (raw_img[y][x][0] / 255)
                            i += 1
                    # In the training data, all images are centered. Using not centered test data will
                    # cause a lot of mislabels due to the nature of perceptron networks
                    # Therefore, we find the center of the pixels from the drawing and try to center it
                    # according to the center of the canvas
                    centered_img = np.zeros((28, 28), dtype='int')
                    i = 0
                    for val in fixed_img:
                        centered_img[int(i/28)][i % 28] = round(val)
                        i += 1
                    center_x = 0
                    center_y = 0
                    points = []
                    for y in range(28):
                        for x in range(28):
                            if not centered_img[x][y] == 0:
                                center_x += x
                                center_y += y
                                points.append((x, y))
                    center_x /= len(points)
                    center_y /= len(points)
                    shift_x = center_x - 13.5
                    shift_y = center_y - 13.5
                    centered_img = np.zeros((28, 28), dtype='int')
                    for x, y in points:
                        if -1 < x - shift_x < 28 and -1 < y - shift_y < 28:
                            centered_img[int(x - shift_x)][int(y - shift_y)] = 1
                    i = 0
                    # Transferring the newly centered image back into a single array of numbers to be run through the NN
                    for arr in centered_img:
                        for val in arr:
                            fixed_img[i] = val
                            i += 1

                    guess = []
                    i = 0
                    # printing the full ordered list of guesses to the console
                    for estimate in number_NN.get_output(fixed_img):
                        guess.append((i, estimate[0]))
                        i += 1
                    guess.sort(key=lambda x: x[1])
                    for num in guess:
                        print(num)
                    print()
                    # outputting the top 3 guesses to the pygames screen
                    for j in range(3):
                        confidence = truncate(guess[-j-1][1], 3)
                        g_txt = str(guess[-j-1][0]) + ", " + str(confidence)
                        guess_text[j][0] = font.render(g_txt, True, (0, 0, 0), (200, 200, 200))
                    for txt in guess_text:
                        window.blit(txt[0], txt[1])

            if event.type == pygame.MOUSEBUTTONUP:
                drawing = False

        if drawing:
            # if user is clicking above the menu at the bottom, fill in the pixels at the mouse
            if pygame.mouse.get_pos()[1] < scale * 27:
                mouse_xy = pygame.mouse.get_pos()
                mouse_x = int(mouse_xy[0] / scale) * scale
                mouse_y = int(mouse_xy[1] / scale) * scale
                pygame.draw.rect(window, (255, 255, 255), (mouse_x - scale, mouse_y - scale, scale * 3, scale * 3))
                # redraw the grid when it is drawn on
                for x in range(28):
                    pygame.draw.rect(window, (55, 55, 55), (x * scale, 0, 1, scale * 28))
                for y in range(29):
                    pygame.draw.rect(window, (55, 55, 55), (0, y * scale, scale * 28, 1))

        pygame.display.update()

    pygame.quit()


def truncate(value, decimals):
    value = value * (10**decimals)
    value = int(value)
    return value / (10**decimals)


def sig(x):
    return 1 / (1 + np.exp(-x))


def sig_derivative(x):
    return sig(x) * (1 - sig(x))


class SGDNN():
    # layers input as an array of neurons per layer such as [3, 5, 5, 1]
    # the first layer is assumed to be the input layer
    def __init__(self, layers):
        self.layers = layers
        self.weights = [(np.random.rand(r, c) * 2 - 1) for r, c in zip(self.layers[1:], self.layers[:-1])]
        self.biases = [np.zeros((c, 1), dtype=float) for c in self.layers[1:]]

    # input_data should be a 2d array with an input to be tested at each index
    def get_output(self, input_data):
        data = np.transpose([input_data])
        for i in range(0, len(self.weights)):
            data = sig(np.dot(self.weights[i], data) + self.biases[i])
        return data

    def train_network(self, input_data, y, iterations=1000, learning_rate=1, batch_size=-1):
        # Getting a batch of the size requested and formating it to be used to train the network
        if not batch_size == -1 and batch_size < len(input_data):
            batch = np.random.randint(0, len(train_number), size=batch_size)
            mini_input = [input_data[i] for i in batch]
            mini_output = [y[i] for i in batch]
            input_data = mini_input
            y = mini_output
        # get the raw outputs and activations of the neurons for each input
        for run in range(iterations):
            print(run)
            delta_w = [np.zeros(np.shape(w)) for w in self.weights]
            delta_b = [np.zeros(np.shape(b)) for b in self.biases]
            for trial, result in zip(input_data, y):
                grad_w, grad_b = self.calculate_grad(trial, result)
                delta_w = [dw + gw for dw, gw in zip(delta_w, grad_w)]
                delta_b = [dw + gw for dw, gw in zip(delta_b, grad_b)]
            for i in range(len(self.weights)):
                self.weights[i] += (learning_rate * delta_w[i] / len(input_data))
                self.biases[i] += (learning_rate * delta_b[i] / len(input_data))

    # calculating gradients
    def calculate_grad(self, input_data, y):
        output = np.transpose([input_data])
        outputs = [output]
        sigmoid_out = np.transpose([input_data])
        sigmoid_outs = [output]
        for w, b in zip(self.weights, self.biases):
            output = np.dot(w, sigmoid_out) + b
            outputs.append(output)
            sigmoid_out = sig(output)
            sigmoid_outs.append(sigmoid_out)
        deriv_error = [2 * (actual - prediction) for actual, prediction in zip(y, sigmoid_outs[-1])]
        delta = deriv_error * sig_derivative(outputs[-1])
        del_w = [np.zeros(np.shape(w)) for w in self.weights]
        del_b = [np.zeros(np.shape(b)) for b in self.biases]
        del_w[-1] = np.dot(delta, np.transpose(sigmoid_outs[-2]))
        del_b[-1] = delta
        for i in range(2, len(self.layers)):
            out = outputs[-i]
            dir_sig = sig_derivative(out)
            delta = np.dot(np.transpose(self.weights[-i + 1]), delta) * dir_sig
            del_w[-i] = np.dot(delta, np.transpose(outputs[-i - 1]))
            del_b[-i] = delta
        return del_w, del_b

    # load the weights given into the neural network to allow performing gradient decent
    # across multiple sessions
    # assumes the loaded weights are a single 1-d array of floating point values with as many values as
    # there are total weights in the neural network
    def load_weights(self, load_weights):
        i = 0
        for layer in range(len(self.weights)):
            for row in range(len(self.weights[layer])):
                for col in range(len(self.weights[layer][row])):
                    self.weights[layer][row][col] = load_weights[i]
                    i += 1


use_saved_weights = True
train_network = False
number_NN = SGDNN([784, 16, 16, 10])
# loads the weights and creates the NN with the loaded weights
if use_saved_weights:
    loaded_weights = np.loadtxt("sweights.txt", delimiter=',')
    number_NN.load_weights(loaded_weights)

# perform gradient decent "iterations" times to try and better optimize the NN
if train_network:
    train_number = np.loadtxt("mnist_test.csv", delimiter=",")
    numbers_input = []
    numbers_output = []
    for i in range(len(train_number)):
        num_expect = np.zeros(10, dtype=int)
        num_expect[int(train_number[i][0])] = 1
        numbers_fixed = np.zeros(784, dtype=float)
        for j in range(1, 785):
            numbers_fixed[j - 1] = train_number[i][j] / 255
        numbers_output.append(num_expect)
        numbers_input.append(numbers_fixed)

    number_NN.train_network(numbers_input, numbers_output, iterations=100, learning_rate=1.5)
    print(number_NN.get_output(numbers_input[3]))
    print(numbers_output[3])
    print(number_NN.get_output(numbers_input[6]))
    print(numbers_output[6])
    print(number_NN.get_output(numbers_input[1]))
    print(numbers_output[1])

boot_display()

# After code completes, save the weights to a text file to be loaded later
saved_weights = open("sweights.txt", "w")
first_value = True
for layer in number_NN.weights:
    for neuron in layer:
        for w in neuron:
            if not first_value:
                saved_weights.write(",")
                saved_weights.write(str(w))
            else:
                saved_weights.write(str(w))
                first_value = False
