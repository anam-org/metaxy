<script setup lang="ts">
defineProps<{
  label: string
  name: string
}>()
</script>

<template>
  <div class="stage-card">
    <div class="mono-label text-teal-300">{{ label }}</div>
    <div class="j-serif mt-2 text-base leading-snug text-white flex-1">{{ name }}</div>
    <slot />
  </div>
</template>
