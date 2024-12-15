import pygame
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

MIN_TRANSFORMERS_VERSION = '4.25.1'
assert transformers.__version__ >= MIN_TRANSFORMERS_VERSION, f'Please upgrade transformers to version {MIN_TRANSFORMERS_VERSION} or higher.'

tokenizer = AutoTokenizer.from_pretrained("Waterhorse/chessgpt-chat-v1")
model = AutoModelForCausalLM.from_pretrained("Waterhorse/chessgpt-chat-v1", torch_dtype=torch.float16)
model = model.to('cuda:0')

pygame.init()

WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
FONT_SIZE = 24
BG_COLOR = (30, 30, 30)
TEXT_COLOR = (200, 200, 200)
INPUT_COLOR = (50, 50, 50)
INPUT_TEXT_COLOR = (255, 255, 255)
CURSOR_COLOR = (255, 255, 255)
PADDING = 10
LINE_SPACING = 5
CURSOR_BLINK_INTERVAL = 500  

FONT = pygame.font.Font(None, FONT_SIZE)

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Chatbot")
clock = pygame.time.Clock()

input_box = pygame.Rect(PADDING, WINDOW_HEIGHT - 50, WINDOW_WIDTH - 2 * PADDING, 40)
input_text = ""
chat_history = []
cursor_visible = True 
last_cursor_blink_time = pygame.time.get_ticks()

def chatbot_response(prompt):
    """Get the chatbot's response to a prompt."""
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    input_length = inputs.input_ids.shape[1]
    outputs = model.generate(
        **inputs, max_new_tokens=128, do_sample=True, temperature=0.7, top_p=0.7, top_k=50, return_dict_in_generate=True,
    )
    token = outputs.sequences[0, input_length:]
    return tokenizer.decode(token)

def wrap_text(text, font, max_width):
    """Wrap text into lines that fit within the specified width."""
    words = text.split(' ')
    lines = []
    current_line = ""

    for word in words:
        test_line = f"{current_line} {word}".strip()
        if font.size(test_line)[0] <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)

    return lines

def draw_chat_window():
    """Draw the chat window and text input area."""
    global cursor_visible, last_cursor_blink_time

    current_time = pygame.time.get_ticks()
    if current_time - last_cursor_blink_time >= CURSOR_BLINK_INTERVAL:
        cursor_visible = not cursor_visible
        last_cursor_blink_time = current_time

    screen.fill(BG_COLOR)

    y_offset = PADDING
    for line in chat_history[-20:]: 
        wrapped_lines = wrap_text(line, FONT, WINDOW_WIDTH - 2 * PADDING)
        for wrapped_line in wrapped_lines:
            text_surface = FONT.render(wrapped_line, True, TEXT_COLOR)
            screen.blit(text_surface, (PADDING, y_offset))
            y_offset += FONT_SIZE + LINE_SPACING

    pygame.draw.rect(screen, INPUT_COLOR, input_box)
    input_surface = FONT.render(input_text, True, INPUT_TEXT_COLOR)
    screen.blit(input_surface, (input_box.x + PADDING, input_box.y + (input_box.height - FONT_SIZE) // 2))

    if cursor_visible:
        cursor_x = input_box.x + PADDING + FONT.size(input_text)[0]
        cursor_y = input_box.y + (input_box.height - FONT_SIZE) // 2
        pygame.draw.line(screen, CURSOR_COLOR, (cursor_x, cursor_y), (cursor_x, cursor_y + FONT_SIZE), 2)

    pygame.display.flip()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN: 
                if input_text.strip():
                    user_query = f"You: {input_text.strip()}"
                    chat_history.append(user_query)

                    bot_reply = chatbot_response(input_text.strip())
                    chat_history.append(f"Bot: {bot_reply}")

                    input_text = ""
            elif event.key == pygame.K_BACKSPACE: 
                input_text = input_text[:-1]
            else:
                input_text += event.unicode

    draw_chat_window()
    clock.tick(30)

pygame.quit()
