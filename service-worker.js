/**
 * Service Worker - Enable offline functionality
 * Install by registering in your main app
 */

const CACHE_NAME = 'mnist-v1';
const URLS_TO_CACHE = [
    '/',
    '/index.html',
    '/frontend_state.js',
    '/styles.css',
];

// Install event - cache resources
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => {
            console.log('Service Worker: Caching files');
            return cache.addAll(URLS_TO_CACHE).catch((err) => {
                console.warn('Service Worker: Some files failed to cache', err);
            });
        })
    );
    self.skipWaiting();
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((cacheNames) => {
            return Promise.all(
                cacheNames.map((cacheName) => {
                    if (cacheName !== CACHE_NAME) {
                        console.log('Service Worker: Deleting old cache:', cacheName);
                        return caches.delete(cacheName);
                    }
                })
            );
        })
    );
    self.clients.claim();
});

// Fetch event - serve from cache, fallback to network
self.addEventListener('fetch', (event) => {
    // Only handle GET requests
    if (event.request.method !== 'GET') {
        return;
    }

    // Skip non-http(s) requests
    if (!event.request.url.startsWith('http')) {
        return;
    }

    event.respondWith(
        // Strategy: Network first, fallback to cache
        fetch(event.request)
            .then((response) => {
                // Don't cache if not successful
                if (!response || response.status !== 200 || response.type !== 'basic') {
                    return response;
                }

                // Clone and cache successful responses
                const responseToCache = response.clone();
                caches.open(CACHE_NAME).then((cache) => {
                    cache.put(event.request, responseToCache);
                });

                return response;
            })
            .catch(() => {
                // Network failed, try cache
                return caches.match(event.request).then((response) => {
                    if (response) {
                        return response;
                    }

                    // Return offline page if available
                    if (event.request.destination === 'document') {
                        return caches.match('/index.html');
                    }

                    return new Response('Offline - Resource not available', {
                        status: 503,
                        statusText: 'Service Unavailable',
                        headers: new Headers({
                            'Content-Type': 'text/plain'
                        })
                    });
                });
            })
    );
});

// Background sync for offline predictions
self.addEventListener('sync', (event) => {
    if (event.tag === 'sync-predictions') {
        event.waitUntil(
            // Queue predictions to sync when back online
            self.registration.showNotification('MNIST', {
                body: 'Syncing predictions with server...',
                icon: '📊'
            })
        );
    }
});

// Handle messages from the app
self.addEventListener('message', (event) => {
    if (event.data && event.data.type === 'SKIP_WAITING') {
        self.skipWaiting();
    }

    if (event.data && event.data.type === 'CLEAR_CACHE') {
        caches.delete(CACHE_NAME).then(() => {
            console.log('Service Worker: Cache cleared');
        });
    }
});
