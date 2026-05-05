import json
import pathlib
import datetime
import html
from IPython.display import display
import ipywidgets as widgets

class AnnotationUI:
    def __init__(self, eval_dataset_path, validation_dataset_path, mappings_file,
                 auto_classifier=None):
        self.eval_dataset_path = pathlib.Path(eval_dataset_path)
        self.validation_dataset_path = pathlib.Path(validation_dataset_path)
        self.mappings_file = pathlib.Path(mappings_file)
        self.auto_classifier = auto_classifier  # Optional SessionAutoClassifier

        # State variables
        self.current_session_idx = 0
        self.current_session_data = None
        self.current_category = 'Unmapped'
        self.current_turn_page = 0
        self.turns_per_page = 5
        self._is_updating = False 

        # Load datasets
        self._load_datasets()

        # Create widgets and connect handlers
        if self.eval_records:
            self._create_widgets()
            self._connect_handlers()

    def _load_datasets(self):
        self.eval_records = []
        if self.eval_dataset_path.exists():
            try:
                self.eval_records = [json.loads(line) for line in self.eval_dataset_path.read_text().splitlines() if line.strip()]
            except Exception: pass

        val_records = []
        if self.validation_dataset_path.exists():
            try:
                val_records = [json.loads(line) for line in self.validation_dataset_path.read_text().splitlines() if line.strip()]
            except Exception: pass

        self.val_by_category = {}
        for rec in val_records:
            category = rec.get('category', 'Unknown')
            if category not in self.val_by_category:
                self.val_by_category[category] = []
            self.val_by_category[category].append(rec)

    def _create_widgets(self):
        # Session Selector
        self.session_selector = widgets.Dropdown(
            options=[(f"{i+1}. {r.get('sessionId', 'Unknown')[:30]}...", i) for i, r in enumerate(self.eval_records)],
            value=0, description='Session:', layout=widgets.Layout(width='450px')
        )

        # ========================================================================
        # The Container Pattern to support Scrolling
        # ========================================================================
        
        # 1. Defines the scrollable container style
        scroll_layout = widgets.Layout(
            border='1px solid #ccc', 
            padding='10px', 
            margin='10px 0', 
            height='500px',        # The container stays 500px
            overflow_y='scroll',   # The container scrolls
            width='98%'
        )

        # 2. Create the inner HTML widgets (these hold the actual text)
        # We set them to auto height so they grow as much as they need
        self.eval_html_content = widgets.HTML(value="", layout=widgets.Layout(width='98%'))
        self.val_html_content = widgets.HTML(value="", layout=widgets.Layout(width='98%'))

        # 3. Create the outer Containers (VBox) that force the scrollbar
        self.eval_container = widgets.VBox([self.eval_html_content], layout=scroll_layout)
        self.val_container = widgets.VBox([self.val_html_content], layout=scroll_layout)

        # Controls
        available_categories = list(self.val_by_category.keys()) + ['Other', 'Unmapped']
        self.category_selector = widgets.Dropdown(
            options=available_categories, value='Unmapped', description='Category:', layout=widgets.Layout(width='300px')
        )

        self.save_button = widgets.Button(description='💾 Save', button_style='success', layout=widgets.Layout(width='100px'))
        self.status_label = widgets.HTML(value='<i>Select a session to begin</i>')
        self.auto_label = widgets.HTML(value='')  # shows auto-classification suggestion
        
        self.prev_button = widgets.Button(description='⬅️ Previous Session', layout=widgets.Layout(width='150px'))
        self.next_button = widgets.Button(description='Next Session ➡️', layout=widgets.Layout(width='150px'))
        
        self.prev_turn_page_button = widgets.Button(description='⬅️ Prev Turns', layout=widgets.Layout(width='120px'))
        self.next_turn_page_button = widgets.Button(description='Next Turns ➡️', layout=widgets.Layout(width='120px'))
        self.turn_page_label = widgets.HTML(value='')

    def _connect_handlers(self):
        self.session_selector.observe(self._on_session_change, names='value')
        self.category_selector.observe(self._on_category_change, names='value')
        self.save_button.on_click(self._on_save_click)
        self.prev_button.on_click(self._on_prev_click)
        self.next_button.on_click(self._on_next_click)
        self.prev_turn_page_button.on_click(self._on_prev_turn_page_click)
        self.next_turn_page_button.on_click(self._on_next_turn_page_click)

    # ... [Helper methods] ...
    def _load_mappings(self):
        if self.mappings_file.exists(): return json.loads(self.mappings_file.read_text())
        return []

    def _save_mapping(self, session_id, category):
        mappings = self._load_mappings()
        mappings = [m for m in mappings if m["sessionId"] != session_id]
        mappings.append({"sessionId": session_id, "category": category, "timestamp": datetime.datetime.now().isoformat()})
        self.mappings_file.parent.mkdir(parents=True, exist_ok=True)
        self.mappings_file.write_text(json.dumps(mappings, indent=2))

    def _get_mapping(self, session_id):
        for m in self._load_mappings():
            if m["sessionId"] == session_id: return m["category"]
        return None

    @staticmethod
    def _parse_turn(turn):
        role_keys = ['user', 'assistant', 'systemPrompt', 'tools']
        for key in turn.keys():
            if key in role_keys: return key, turn[key]
        return 'unknown', str(turn)

    def _get_display_turns(self):
        if not self.current_session_data: return []
        turns = self.current_session_data.get('turns', [])
        display_turns = []
        for turn in turns:
            role, content = self._parse_turn(turn)
            if role != 'systemPrompt':
                display_turns.append((role, content))
        return display_turns

    # ============================================================================
    # Display Logic - UPDATING THE INNER HTML WIDGET
    # ============================================================================

    def _display_eval_session(self):
        if not self.current_session_data: return

        display_turns = self._get_display_turns()
        
        # Pagination
        start_idx = self.current_turn_page * self.turns_per_page
        end_idx = min(start_idx + self.turns_per_page, len(display_turns))
        total_pages = (len(display_turns) + self.turns_per_page - 1) // self.turns_per_page
        self.turn_page_label.value = f'Page {self.current_turn_page + 1} of {total_pages}'

        # Build HTML String
        html_parts = []
        html_parts.append(f"<div style='font-family: monospace;'>")
        
        for idx in range(start_idx, end_idx):
            role, content = display_turns[idx]
            
            if role == 'tools' and isinstance(content, (dict, list)):
                content_str = json.dumps(content, indent=2)
            else:
                content_str = str(content)
            
            safe_content = html.escape(content_str)
            
            html_parts.append(f"<strong>Turn {idx + 1} - {role.upper()}:</strong>")
            html_parts.append(f"<pre style='white-space: pre-wrap; word-wrap: break-word; margin: 5px 0 20px 0;'>{safe_content}</pre>")
            html_parts.append("<hr style='border-top: 1px dashed #ccc;'>")
            
        html_parts.append("</div>")
        
        # UPDATE THE INNER CONTENT
        self.eval_html_content.value = "".join(html_parts)

    def _display_validation_examples(self):
        if self.current_category not in self.val_by_category:
            self.val_html_content.value = f"<i>No validation examples found for: {self.current_category}</i>"
            return
            
        examples = self.val_by_category[self.current_category]
        
        html_parts = []
        html_parts.append(f"<strong>VALIDATION EXAMPLES: {self.current_category}</strong><hr>")
        html_parts.append("<div style='font-family: monospace;'>")

        for example in examples[:1]:
            turns = example.get('turns', [])
            displayed_count = 0
            for turn in turns:
                if displayed_count >= 5: break
                role, content = self._parse_turn(turn)
                if role == 'systemPrompt': continue
                
                displayed_count += 1
                safe_content = html.escape(str(content)[:200]) + "..."
                
                html_parts.append(f"<strong>{role.upper()}:</strong>")
                html_parts.append(f"<pre style='white-space: pre-wrap; word-wrap: break-word; margin: 2px 0 10px 0; color: #555;'>{safe_content}</pre>")
        
        html_parts.append("</div>")
        
        # UPDATE THE INNER CONTENT
        self.val_html_content.value = "".join(html_parts)

    def _update_display(self):
        session_id = self.current_session_data.get('sessionId', '') if self.current_session_data else ''
        saved_category = self._get_mapping(session_id)

        # Check for span-injected category (Option A — set by test runners)
        span_category = self.current_session_data.get('category') if self.current_session_data else None

        self._is_updating = True
        if saved_category:
            self.category_selector.value = saved_category
            self.current_category = saved_category
            self.status_label.value = f'<span style="color: blue;">✓ Existing mapping: {saved_category}</span>'
        elif span_category and span_category in list(self.val_by_category.keys()) + ['Other', 'Unmapped']:
            # Auto-apply span metadata category — no human needed
            self.category_selector.value = span_category
            self.current_category = span_category
            self.status_label.value = (
                f'<span style="color: #2e7d32;">🏷️ Auto-applied from span metadata: {span_category}</span>'
            )
        else:
            self.category_selector.value = 'Unmapped'
            self.current_category = 'Unmapped'
            self.status_label.value = '<i>No mapping saved for this session</i>'
        self._is_updating = False

        # Show LLM auto-classification suggestion (Option B)
        self.auto_label.value = ''
        if self.auto_classifier and self.current_session_data and not saved_category and not span_category:
            try:
                suggestion = self.auto_classifier.classify(self.current_session_data)
                if suggestion:
                    self.auto_label.value = (
                        f'<span style="color: #e65100;">🤖 LLM suggestion: <b>{suggestion}</b> '
                        f'— <a href="#" onclick="return false;" '
                        f'style="color:#1565c0;">click category dropdown to accept or override</a></span>'
                    )
                    # Pre-select the suggestion in the dropdown for one-click acceptance
                    if suggestion in list(self.val_by_category.keys()) + ['Other', 'Unmapped']:
                        self._is_updating = True
                        self.category_selector.value = suggestion
                        self.current_category = suggestion
                        self._is_updating = False
            except Exception:
                pass

        self._display_eval_session()
        self._display_validation_examples()

    # ============================================================================
    # Event Handlers
    # ============================================================================

    def _on_session_change(self, change):
        if self._is_updating: return
        self.current_session_idx = change['new']
        self.current_session_data = self.eval_records[self.current_session_idx]
        self.current_turn_page = 0
        self._update_display()

    def _on_category_change(self, change):
        if self._is_updating: return
        self.current_category = change['new']
        self._display_validation_examples()

    def _on_save_click(self, b): 
        if self.current_session_data:
            self._save_mapping(self.current_session_data.get('sessionId'), self.category_selector.value)
            self.status_label.value = f'<span style="color: green;">✅ Saved: {self.category_selector.value}</span>'

    def _on_prev_click(self, b): 
        if self.session_selector.value > 0: self.session_selector.value -= 1
    def _on_next_click(self, b): 
        if self.session_selector.value < len(self.eval_records) - 1: self.session_selector.value += 1
    def _on_prev_turn_page_click(self, b):
        if self.current_turn_page > 0:
            self.current_turn_page -= 1
            self._display_eval_session()
    def _on_next_turn_page_click(self, b):
        display_turns = self._get_display_turns()
        total_pages = (len(display_turns) + self.turns_per_page - 1) // self.turns_per_page
        if self.current_turn_page < total_pages - 1:
            self.current_turn_page += 1
            self._display_eval_session()

    def display(self):
        if not self.eval_records:
            print("No records found")
            return
        
        top_controls = widgets.HBox([self.session_selector, self.category_selector, self.save_button])
        nav_controls = widgets.HBox([self.prev_button, self.next_button])
        turn_controls = widgets.HBox([self.prev_turn_page_button, self.turn_page_label, self.next_turn_page_button])

        # Display the CONTAINERS (eval_container), not the raw HTML content
        ui_box = widgets.VBox([
            widgets.HTML('<h3>Annotation Tool</h3>'),
            top_controls, nav_controls, self.status_label, self.auto_label,
            widgets.HTML('<h4>Eval Session:</h4>'), turn_controls, self.eval_container,
            widgets.HTML('<h4>Validation:</h4>'), self.val_container
        ])

        self.current_session_data = self.eval_records[0]
        self._update_display()
        display(ui_box)