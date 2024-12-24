class Table:
        def __init__(self, *, title=None, headers=None, contents=None):
            self.title = title if title is not None else ''
            self.headers = headers if headers is not None else ['', '']
            self.contents = contents if contents is not None else {}
            self.key_width = max([len(self.headers[0]), *[len(str(key)) for key in self.contents.keys()]])
            self.value_width = max([len(self.headers[1]), *[len(str(value)) for value in self.contents.values()]])
        def display_title(self):
            print(f'{self.title:^{self.key_width + self.value_width + 7}}')
        def display_line(self, symbols):
            start, line, middle, end = symbols
            print(start + line * (self.key_width + 2) + middle + line * (self.value_width + 2) + end)
        def display_header(self):
            print(f'│ {str(self.headers[0]):^{self.key_width}} │ {str(self.headers[1]):^{self.value_width}} │')
        def display_content(self, key, value):
            print(f'│ {str(key):<{self.key_width}} │ {str(value):<{self.value_width}} │')
        def display(self):
            if self.title == '' and self.headers == ['', ''] and len(self.contents) == 0:
                return
            if self.title != '':
                self.display_title()
            self.display_line('┌─┬┐')
            if self.headers != ['', '']:
                self.display_header()
                if len(self.contents) != 0:
                    self.display_line('╞═╪╡')
            for i, (key, value) in enumerate(self.contents.items()):
                self.display_content(key, value)
                if i + 1 != len(self.contents):
                    self.display_line('├─┼┤')
            self.display_line('└─┴┘')
