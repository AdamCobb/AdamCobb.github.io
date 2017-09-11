---
layout: page
title: Blog
---
***
<ul class="posts">
  {% for post in site.posts %}

    <h1>
    <a href="{{ site.github.url }}{{ post.url }}">{{ post.title }}</a>
  </h1>
  {% if post.image.teaser %}
    <a href="{{ site.github.url }}{{ post.url }}"><img src="{{ site.github.url }}/images/{{ post.image.teaser }}"></a>
  {% endif %}
  <p>
    {{ post.content | strip_html | truncate: 350 }} <a href="{{ site.github.url }}{{ post.url }}">Read more</a>
    <span class="post-date" style="margin-top:3px"><i class="fa fa-calendar" aria-hidden="true"></i> {{ post.date | date_to_string }} - <i class="fa fa-clock-o" aria-hidden="true"></i> {% include read-time.html %}</span>
  </p>

  {% endfor %}
</ul>