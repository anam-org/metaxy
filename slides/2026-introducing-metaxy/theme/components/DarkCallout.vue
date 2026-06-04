<script setup lang="ts">
defineProps<{
  label: string
}>()
</script>

<template>
  <div class="dark-callout">
    <div class="mono-label text-teal-200">{{ label }}</div>
    <slot />
  </div>
</template>
