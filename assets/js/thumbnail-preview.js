function fetchOGTags(url) {
  return fetch(`https://cors-anywhere.herokuapp.com/${url}`)
    .then(response => response.text())
    .then(html => {
      const parser = new DOMParser();
      const doc = parser.parseFromString(html, 'text/html');
      const title = doc.querySelector('meta[property="og:title"]').content;
      const image = doc.querySelector('meta[property="og:image"]').content;
      return { title, image };
    })
    .catch(error => console.error('Error fetching OG tags:', error));
}

function displayThumbnailPreview(url, containerId) {
  fetchOGTags(url).then(({ title, image }) => {
    const container = document.getElementById(containerId);
    if (container) {
      container.innerHTML = `
        <a href="${url}" target="_blank">
          <img src="${image}" alt="${title}" />
          <h4>${title}</h4>
        </a>`;
    }
  });
}

// Replace this URL with the URL of the external article
const externalArticleUrl = 'https://www.sri.com/story/solving-unsolvable-math-challenges-with-quantum-inspired-computers';

// Replace this ID with the ID of the placeholder element in your post/page
const previewContainerId = 'external-article-preview';

displayThumbnailPreview(externalArticleUrl, previewContainerId);
