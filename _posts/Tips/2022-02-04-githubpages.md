---
title:  "The shortest intro of GitHub Pages"
categories:
  - Tips-blog
tags:
  - Tips
  - GitHub Pages
  - Jekyll
---

GitHub Pages: Static website (blog) hosting service

Jekyll: Static website generator, written in Ruby. Converts markdown files into html files.


To begin:
1. Install Ruby and Jekyll
2. Download a Jekyll theme you like (a blog template)
3. Install the rest files using bundle command
4. Test your website locally
- type "bundle exec jekyll serve" in the package directory
- type "localhost:4000/" at your browser
5. Create a git repo ("username.github.io") at GitHub website
6. git push your code to remote repo
7. Check out "https://username.github.io"


Posting process:
0. Create _posts directory at your package directory if you don't have it yet.
1. Go to _posts directory and write a post as a markdown file
2. Test your code and see website preview from your local computer
3. git push your final file(s) to your remote git repo
4. Then GitHub pages automatically runs Jekyll to update your website

Note:
_config.yml file is read only once at build (bundle command at your local computer)