#!/usr/bin/python3
from fastapi import FastAPI
from typing import Optional

api = FastAPI()

all_todos = [
    {'todo_id': 1, 'todo_name': 'Sports', 'todo_description': 'Go to the gym'},
    {'todo_id': 2, 'todo_name': 'Reads', 'todo_description': 'Read 10 pages'},
    {'todo_id': 3, 'todo_name': 'Shop', 'todo_description': 'Go shopping'},
    {'todo_id': 4, 'todo_name': 'Study', 'todo_description': 'Study for exam'},
    {'todo_id': 5, 'todo_name': 'Meditate', 'todo_description': 'Meditate 20 minutes'},
]

# Root endpoint
@api.get('/')
def index():
    return {"message": "Hello World"}

# Get specific todo by ID
@api.get('/todo/{todo_id}')
def get_todo_by_id(todo_id: int):
    for todo in all_todos:
        if todo['todo_id'] == todo_id:
            return todo
    return {"error": "Todo not found"}

# Get all todos with optional limit
@api.get('/todos')
def get_todos(first_n: Optional[int] = None):
    if first_n:
        return all_todos[:first_n]
    return all_todos  # Fixed typo: was all_todos in some places

# Create new todo
@api.post('/todos')
def create_todo(todo: dict):
    new_todo_id = max(todo['todo_id'] for todo in all_todos) + 1
    new_todo = {
        'todo_id': new_todo_id,
        'todo_name': todo['todo_name'],
        'todo_description': todo['todo_description']
    }
    all_todos.append(new_todo)  # Fixed typo: was all_todos in some places
    return new_todo

# Update todo
@api.put('/todos/{todo_id}')
def update_todo(todo_id: int, updated_todo: dict):  # Fixed typo in function name (was uodate_tod)
    for todo in all_todos:
        if todo['todo_id'] == todo_id:
            todo['todo_name'] = updated_todo['todo_name']
            todo['todo_description'] = updated_todo['todo_description']
            return todo
    return {"error": "Todo not found"}

# Delete todo
@api.delete('/todos/{todo_id}')
def delete_todo(todo_id: int):
    for index, todo in enumerate(all_todos):
        if todo['todo_id'] == todo_id:
            deleted_todo = all_todos.pop(index)  # Fixed typo: was all_todos in some places
            return deleted_todo
    return {"error": "Todo not found"}